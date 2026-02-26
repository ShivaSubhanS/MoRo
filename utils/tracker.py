from ultralytics import YOLO

import torch
from tqdm import tqdm
from collections import defaultdict

from utils.demo_utils import get_video_lwh

class Tracker:
    def __init__(self) -> None:
        # https://docs.ultralytics.com/modes/predict/
        # Use yolo11n (nano) on CPU to keep GPU fully free for the main model
        self.yolo = YOLO("yolo26s.pt")

    def track(self, image_paths, chunk_size=500):
        cfg = {
            "device": "cpu",  # run on CPU to keep GPU free for main model
            "conf": 0.5,
            "classes": 0,  # human
            "verbose": False,
            "stream": True,
            "persist": True,
            #"imgsz": 640,
        }
        
        track_history = []
        total_frames = len(image_paths)
        
        # Process in chunks
        for chunk_start in range(0, total_frames, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_frames)
            chunk_paths = image_paths[chunk_start:chunk_end]
            
            results = self.yolo.track(chunk_paths, **cfg)
            
            for result in tqdm(results, total=len(chunk_paths), desc=f"Yolo11 Tracking [{chunk_start}:{chunk_end}]"):
                if result.boxes.id is not None:
                    track_ids = result.boxes.id.int().cpu().tolist()  # (N)
                    bbx_xyxy = result.boxes.xyxy.cpu()  # (N, 4)
                    result_frame = [{"id": track_ids[i], "bbx_xyxy": bbx_xyxy[i]} for i in range(len(track_ids))]
                else:
                    result_frame = []
                track_history.append(result_frame)
            torch.cuda.empty_cache()

        del self.yolo  # free up memory
        return track_history

    @staticmethod
    def sort_track_length(track_history, image_paths):
        """This handles the track history from YOLO tracker."""
        id_to_frame_ids = defaultdict(list)
        id_to_bbx_xyxys = defaultdict(list)
        # parse to {det_id : [frame_id]}
        for frame_id, frame in enumerate(track_history):
            for det in frame:
                id_to_frame_ids[det["id"]].append(frame_id)
                id_to_bbx_xyxys[det["id"]].append(det["bbx_xyxy"])
        for k, v in id_to_bbx_xyxys.items():
            id_to_bbx_xyxys[k] = torch.stack(v) 

        # Sort by length of each track (max to min)
        id_length = {k: len(v) for k, v in id_to_frame_ids.items()}
        id2length = dict(sorted(id_length.items(), key=lambda item: item[1], reverse=True))

        # Sort by area sum (max to min)
        id_area_sum = {}
        l, w, h = get_video_lwh(image_paths)
        for k, v in id_to_bbx_xyxys.items():
            bbx_wh = v[:, 2:] - v[:, :2]
            id_area_sum[k] = (bbx_wh[:, 0] * bbx_wh[:, 1] / w / h).sum().item()
        id2area_sum = dict(sorted(id_area_sum.items(), key=lambda item: item[1], reverse=True))
        id_sorted = list(id2area_sum.keys())

        return id_to_frame_ids, id_to_bbx_xyxys, id_sorted

    def get_tracks(self, image_paths, enlarge_factor=1.2, length_threshold=0.6):
        # Track and get all valid tracks
        track_history = self.track(image_paths)
        id_to_frame_ids, id_to_bbx_xyxys, id_sorted = self.sort_track_length(track_history, image_paths)
        # for id, frames in id_to_frame_ids.items():
        #     print(f"Track ID {id}: {len(frames)} frames")
        
        total_frames = get_video_lwh(image_paths)[0]
        min_length = int(total_frames * length_threshold)
        
        # Filter tracks by length threshold
        valid_track_ids = [tid for tid in id_sorted if len(id_to_frame_ids[tid]) >= min_length]
        
        if len(valid_track_ids) == 0:
            raise ValueError(f"No tracks found with length >= {length_threshold * 100}% of total frames")
        else:
            print(f"Found {len(valid_track_ids)} valid tracks with length >= {length_threshold * 100}% of total frames")
        
        # Process each valid track
        all_bbx_xys = []
        
        for track_id in valid_track_ids:
            frame_ids = torch.tensor(id_to_frame_ids[track_id])  # (N,)
            bbx_xyxys = id_to_bbx_xyxys[track_id].float()  # (N, 4)

            # Create output tensor and fill known detections
            bbx_xyxy_one_track = torch.zeros(total_frames, 4)
            bbx_xyxy_one_track[frame_ids] = bbx_xyxys

            # Find missing frame segments and interpolate
            mask = torch.zeros(total_frames, dtype=torch.bool)
            mask[frame_ids] = True
            
            if not mask.all():
                # Find contiguous segments of missing frames
                padded_mask = torch.cat([torch.tensor([True]), mask, torch.tensor([True])])
                diffs = torch.diff(padded_mask.int())
                starts = (diffs == -1).nonzero().flatten()
                ends = (diffs == 1).nonzero().flatten()
                
                # Interpolate each missing segment
                for start, end in zip(starts.tolist(), ends.tolist()):
                    prev_idx, next_idx = start - 1, end
                    
                    if prev_idx < 0:
                        bbx_xyxy_one_track[start:end] = bbx_xyxy_one_track[next_idx]
                    elif next_idx >= total_frames:
                        bbx_xyxy_one_track[start:end] = bbx_xyxy_one_track[prev_idx]
                    else:
                        n = end - start
                        alphas = torch.linspace(0, 1, n + 2)[1:-1, None]
                        bbx_xyxy_one_track[start:end] = bbx_xyxy_one_track[prev_idx] + alphas * (bbx_xyxy_one_track[next_idx] - bbx_xyxy_one_track[prev_idx])

            # bbx_xyxy_one_track = moving_average_smooth(bbx_xyxy_one_track, window_size=5, dim=0)
            # bbx_xyxy_one_track = moving_average_smooth(bbx_xyxy_one_track, window_size=5, dim=0)

            # Convert xyxy to xys (center_x, center_y, scale) and enlarge
            bbx_xys = torch.zeros(total_frames, 3)
            bbx_xys[:, 0] = (bbx_xyxy_one_track[:, 0] + bbx_xyxy_one_track[:, 2]) / 2  # center_x
            bbx_xys[:, 1] = (bbx_xyxy_one_track[:, 1] + bbx_xyxy_one_track[:, 3]) / 2  # center_y
            width = bbx_xyxy_one_track[:, 2] - bbx_xyxy_one_track[:, 0]
            height = bbx_xyxy_one_track[:, 3] - bbx_xyxy_one_track[:, 1]
            bbx_xys[:, 2] = torch.max(width, height) / 200.0 * enlarge_factor  # scale
            
            all_bbx_xys.append(bbx_xys)
        
        # Stack all tracks: [N, L, 3]
        return torch.stack(all_bbx_xys, dim=0)
