import cv2
import numpy as np
import imagehash
from PIL import Image
import os
import librosa
import matplotlib.pyplot as plt
from scipy.spatial.distance import hamming
from collections import defaultdict

class VideoDuplicateDetector:
    def __init__(self, 
                 frame_sample_rate=1,  # Sample 1 frame per second
                 hash_size=16,         # Size of the image hash
                 threshold=0.85):      # Similarity threshold
        self.frame_sample_rate = frame_sample_rate
        self.hash_size = hash_size
        self.threshold = threshold
        self.video_database = defaultdict(dict)  # Store all processed videos
    
    def _extract_frames(self, video_path):
        """Extract frames from video at specified sample rate"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Determine frame sampling interval
        sample_interval = int(fps * self.frame_sample_rate)
        if sample_interval < 1:
            sample_interval = 1
            
        frame_count = 0
        success = True
        
        while success:
            success, frame = cap.read()
            if not success:
                break
                
            if frame_count % sample_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                
            frame_count += 1
            
        cap.release()
        return frames
    
    def _compute_frame_hashes(self, frames):
        """Compute perceptual hashes for each frame"""
        frame_hashes = []
        for frame in frames:
            img = Image.fromarray(frame)
            # Use perceptual hash (phash) which is robust to minor changes
            frame_hash = imagehash.phash(img, hash_size=self.hash_size)
            frame_hashes.append(frame_hash)
        return frame_hashes
    
    def _extract_audio_features(self, video_path):
        """Extract audio fingerprint features from video"""
        # Create a temporary audio file
        temp_audio = "temp_audio.wav"
        os.system(f"ffmpeg -i {video_path} -q:a 0 -map a {temp_audio} -y -loglevel quiet")
        
        try:
            # Load audio and extract MFCC features
            y, sr = librosa.load(temp_audio, sr=None)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            # Get the mean of MFCCs to create a simplified fingerprint
            audio_fingerprint = np.mean(mfccs, axis=1)
            
            # Clean up temporary file
            os.remove(temp_audio)
            return audio_fingerprint
        except:
            # If audio extraction fails, return empty array
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
            return np.array([])
    
    def _compute_motion_signature(self, frames):
        """Create a simplified motion signature for the video"""
        if len(frames) < 2:
            return np.zeros(10)
        
        motion_values = []
        
        for i in range(len(frames) - 1):
            # Convert frames to grayscale
            prev_frame = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            curr_frame = cv2.cvtColor(frames[i+1], cv2.COLOR_RGB2GRAY)
            
            # Calculate optical flow (dense)
            flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
            # Compute the magnitude of flow vectors
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            
            # Get the mean motion magnitude
            motion_values.append(np.mean(magnitude))
        
        # Create a histogram of motion values as signature
        hist, _ = np.histogram(motion_values, bins=10, range=(0, np.max(motion_values) if len(motion_values) > 0 else 1))
        hist = hist / (np.sum(hist) if np.sum(hist) > 0 else 1)  # Normalize
        
        return hist
    
    def _extract_color_histogram(self, frames):
        """Extract color histograms from video frames"""
        histograms = []
        for frame in frames:
            # Convert to HSV color space which is more perceptually meaningful
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            
            # Calculate histograms for each channel
            h_hist = cv2.calcHist([hsv_frame], [0], None, [16], [0, 180])
            s_hist = cv2.calcHist([hsv_frame], [1], None, [16], [0, 256])
            v_hist = cv2.calcHist([hsv_frame], [2], None, [16], [0, 256])
            
            # Normalize histograms
            h_hist = cv2.normalize(h_hist, h_hist, 0, 1, cv2.NORM_MINMAX)
            s_hist = cv2.normalize(s_hist, s_hist, 0, 1, cv2.NORM_MINMAX)
            v_hist = cv2.normalize(v_hist, v_hist, 0, 1, cv2.NORM_MINMAX)
            
            # Concatenate histograms
            hist_features = np.concatenate([h_hist, s_hist, v_hist]).flatten()
            histograms.append(hist_features)
            
        # Return average histogram as a signature
        if histograms:
            return np.mean(np.array(histograms), axis=0)
        return np.array([])
    
    def _detect_scenes(self, frames):
        """Detect scene changes in the video"""
        if len(frames) < 2:
            return []
            
        scene_changes = []
        prev_frame = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
        
        for i in range(1, len(frames)):
            curr_frame = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            
            # Calculate difference between frames
            diff = cv2.absdiff(curr_frame, prev_frame)
            
            # Calculate the percentage of changed pixels
            change_percent = np.count_nonzero(diff > 30) / diff.size
            
            # If change is significant, mark as scene change
            if change_percent > 0.35:  # Threshold can be adjusted
                scene_changes.append(i)
                
            prev_frame = curr_frame
            
        return scene_changes
        
    def _compare_color_histograms(self, hist1, hist2):
        """Compare color histograms using correlation"""
        if hist1.size == 0 or hist2.size == 0:
            return 0.5
            
        similarity = cv2.compareHist(hist1.reshape(-1, 1), 
                                     hist2.reshape(-1, 1), 
                                     cv2.HISTCMP_CORREL)
        # Convert to 0-1 range
        similarity = (similarity + 1) / 2
        return similarity
        
    def _compare_scene_structure(self, scenes1, scenes2, frame_count1, frame_count2):
        """Compare the scene structure between two videos"""
        if not scenes1 or not scenes2:
            return 0.5
            
        # Convert to relative positions
        rel_scenes1 = [s / frame_count1 for s in scenes1]
        rel_scenes2 = [s / frame_count2 for s in scenes2]
        
        # Count how many scenes approximately match
        matches = 0
        for s1 in rel_scenes1:
            for s2 in rel_scenes2:
                if abs(s1 - s2) < 0.05:  # 5% tolerance
                    matches += 1
                    break
                    
        # Similarity based on matching scenes
        max_scenes = max(len(scenes1), len(scenes2))
        if max_scenes == 0:
            return 0.5
            
        return matches / max_scenes
    
    def process_video(self, video_path, video_id=None):
        """Process a video and add it to the database"""
        if video_id is None:
            video_id = os.path.basename(video_path)
        
        # Extract frames
        frames = self._extract_frames(video_path)
        
        if not frames:
            print(f"Could not extract frames from {video_id}")
            return
        
        # Compute features
        frame_hashes = self._compute_frame_hashes(frames)
        audio_features = self._extract_audio_features(video_path)
        motion_signature = self._compute_motion_signature(frames)
        color_histogram = self._extract_color_histogram(frames)
        scene_changes = self._detect_scenes(frames)
        
        # Get video metadata
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = len(frames) / fps if fps > 0 else 0
        cap.release()
        
        # Store in database
        self.video_database[video_id] = {
            'frame_hashes': frame_hashes,
            'audio_features': audio_features,
            'motion_signature': motion_signature,
            'color_histogram': color_histogram,
            'scene_changes': scene_changes,
            'frame_count': len(frames),
            'metadata': {
                'width': width,
                'height': height,
                'fps': fps,
                'duration': duration,
                'aspect_ratio': width / height if height > 0 else 0
            }
        }
        
        return video_id
    
    def _compare_frame_hashes(self, hashes1, hashes2):
        """Compare frame hashes between two videos using dynamic programming"""
        if not hashes1 or not hashes2:
            return 0
        
        # Check if hashes are identical - exact match case
        if len(hashes1) == len(hashes2):
            identical_count = sum(1 for h1, h2 in zip(hashes1, hashes2) if h1 == h2)
            if identical_count == len(hashes1):
                return 1.0
        
        # Handle videos of different lengths
        n, m = len(hashes1), len(hashes2)
        
        # Initialize matrix for dynamic time warping
        dtw = np.zeros((n + 1, m + 1))
        dtw[0, 1:] = float('inf')
        dtw[1:, 0] = float('inf')
        
        # Fill DTW matrix
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                # Calculate hash difference
                hash_diff = 1 - hamming(hashes1[i-1].hash.flatten(), hashes2[j-1].hash.flatten())
                
                cost = 1 - hash_diff  # Convert similarity to cost
                dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
        
        # Normalize by the length of the optimal path
        path_length = n + m
        similarity = 1 - (dtw[n, m] / path_length)
        
        return similarity
    
    def _compare_audio_features(self, audio1, audio2):
        """Compare audio features between two videos"""
        if audio1.size == 0 or audio2.size == 0:
            return 0.5  # Neutral score if audio not available
        
        # Normalize and compute cosine similarity
        norm1 = np.linalg.norm(audio1)
        norm2 = np.linalg.norm(audio2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.5
            
        audio1_normalized = audio1 / norm1
        audio2_normalized = audio2 / norm2
        
        similarity = np.dot(audio1_normalized, audio2_normalized)
        # Adjust to 0-1 range
        similarity = (similarity + 1) / 2
        
        return similarity
    
    def _compare_motion_signatures(self, motion1, motion2):
        """Compare motion signatures between two videos"""
        # Use Bhattacharyya distance for histogram comparison
        similarity = np.sum(np.sqrt(motion1 * motion2))
        return similarity
    
    def check_duplicate(self, query_video_path, return_matches=False):
        """
        Check if a video is a duplicate of any video in the database
        
        Args:
            query_video_path: Path to the video to check
            return_matches: If True, return list of matches with similarity scores
            
        Returns:
            is_duplicate: Boolean indicating if the video is a duplicate
            matches: List of (video_id, similarity) tuples if return_matches=True
        """
        # Check for exact file match first
        query_abs_path = os.path.abspath(query_video_path)
        
        # Process query video
        query_id = "query_" + os.path.basename(query_video_path)
        self.process_video(query_video_path, query_id)
        
        # If database is empty (except for the query we just added)
        if len(self.video_database) <= 1:
            if return_matches:
                return False, []
            return False
        
        query_data = self.video_database[query_id]
        
        # Store matches and their similarity scores
        matches = []
        
        # Compare with all videos in database
        for video_id, video_data in self.video_database.items():
            if video_id == query_id:
                continue
                
            # Check if it's the exact same file path
            db_video_path = video_id
            if os.path.exists(db_video_path) and os.path.abspath(db_video_path) == query_abs_path:
                matches.append((video_id, 1.0))
                continue
                
            # Compare frame hashes with improved thresholding
            frame_similarity = self._compare_frame_hashes(
                query_data['frame_hashes'], 
                video_data['frame_hashes']
            )
            
            # Compare audio features
            audio_similarity = self._compare_audio_features(
                query_data['audio_features'], 
                video_data['audio_features']
            )
            
            # Compare motion signatures
            motion_similarity = self._compare_motion_signatures(
                query_data['motion_signature'], 
                video_data['motion_signature']
            )
            
            # Compare color histograms
            color_similarity = self._compare_color_histograms(
                query_data['color_histogram'],
                video_data['color_histogram']
            )
            
            # Compare scene structures
            scene_similarity = self._compare_scene_structure(
                query_data['scene_changes'],
                video_data['scene_changes'],
                query_data['frame_count'],
                video_data['frame_count']
            )
            
            # Compare metadata (simple check for similar dimensions and duration)
            q_meta = query_data['metadata']
            v_meta = video_data['metadata']
            
            # Check if aspect ratios are close
            aspect_similarity = 1.0 - min(
                abs(q_meta['aspect_ratio'] - v_meta['aspect_ratio']) / max(q_meta['aspect_ratio'], v_meta['aspect_ratio']),
                0.5  # Cap the difference
            ) if q_meta['aspect_ratio'] > 0 and v_meta['aspect_ratio'] > 0 else 0.5
            
            # Check if durations are close
            duration_similarity = 1.0 - min(
                abs(q_meta['duration'] - v_meta['duration']) / max(q_meta['duration'], v_meta['duration']),
                0.5  # Cap the difference
            ) if q_meta['duration'] > 0 and v_meta['duration'] > 0 else 0.5
            
            # Improved weighting system (adjusted weights)
            overall_similarity = (
                0.50 * frame_similarity + 
                0.20 * audio_similarity + 
                0.10 * motion_similarity +
                0.10 * color_similarity +
                0.05 * scene_similarity + 
                0.025 * aspect_similarity +
                0.025 * duration_similarity
            )
            
            # Apply non-linear scaling to boost high similarities even higher
            if overall_similarity > 0.8:
                overall_similarity = 0.8 + (overall_similarity - 0.8) * 1.5
                # Ensure we don't exceed 1.0
                overall_similarity = min(overall_similarity, 1.0)
            
            if overall_similarity >= self.threshold:
                matches.append((video_id, overall_similarity))
        
        # Clean up query from database if it was just for checking
        del self.video_database[query_id]
        
        # Sort matches by similarity score
        matches.sort(key=lambda x: x[1], reverse=True)
        
        is_duplicate = len(matches) > 0
        
        if return_matches:
            return is_duplicate, matches
        return is_duplicate
    
    def visualize_comparison(self, video1_path, video2_path, num_frames=5):
        """
        Visualize the comparison between two videos by showing sampled frames
        and their hash differences
        """
        # Process both videos if not already in database
        video1_id = os.path.basename(video1_path)
        video2_id = os.path.basename(video2_path)
        
        if video1_id not in self.video_database:
            self.process_video(video1_path, video1_id)
        
        if video2_id not in self.video_database:
            self.process_video(video2_path, video2_id)
        
        # Get frames for visualization
        frames1 = self._extract_frames(video1_path)
        frames2 = self._extract_frames(video2_path)
        
        # Sample frames evenly
        if len(frames1) > num_frames:
            indices1 = np.linspace(0, len(frames1)-1, num_frames, dtype=int)
            frames1 = [frames1[i] for i in indices1]
            
        if len(frames2) > num_frames:
            indices2 = np.linspace(0, len(frames2)-1, num_frames, dtype=int)
            frames2 = [frames2[i] for i in indices2]
        
        # Get hash differences
        is_duplicate, matches = self.check_duplicate(video2_path, return_matches=True)
        
        # Calculate overall similarity
        if video1_id in [match[0] for match in matches]:
            similarity = next(match[1] for match in matches if match[0] == video1_id)
        else:
            # Compare directly if not found in matches
            video1_data = self.video_database[video1_id]
            video2_data = self.video_database[video2_id]
            
            frame_similarity = self._compare_frame_hashes(
                video1_data['frame_hashes'], 
                video2_data['frame_hashes']
            )
            
            audio_similarity = self._compare_audio_features(
                video1_data['audio_features'], 
                video2_data['audio_features']
            )
            
            motion_similarity = self._compare_motion_signatures(
                video1_data['motion_signature'], 
                video2_data['motion_signature']
            )
            
            similarity = 0.6 * frame_similarity + 0.25 * audio_similarity + 0.15 * motion_similarity
        
        # Create visualization
        fig, axes = plt.subplots(2, len(frames1), figsize=(15, 6))
        
        # Show frames from both videos
        for i in range(len(frames1)):
            if i < len(frames1):
                axes[0, i].imshow(frames1[i])
                axes[0, i].set_title(f"Video 1: Frame {i}")
                axes[0, i].axis('off')
            
            if i < len(frames2):
                axes[1, i].imshow(frames2[i])
                axes[1, i].set_title(f"Video 2: Frame {i}")
                axes[1, i].axis('off')
            
        plt.suptitle(f"Video Comparison - Similarity: {similarity:.2f}" + 
                    f" ({'Duplicate' if similarity >= self.threshold else 'Not Duplicate'})")
        plt.tight_layout()
        plt.show()


# Example usage
def example_usage():
    detector = VideoDuplicateDetector(threshold=0.75)
    
    # Add videos to database
    video1_path = "./1.mp4"
    detector.process_video(video1_path)
    
    # Second video for comparison
    video2_path = "./1.mp4"  # Can be the same as video1 to test exact duplicates
    
    print("\n===== VIDEO COMPARISON RESULTS =====")
    print(f"Video 1: {video1_path}")
    print(f"Video 2: {video2_path}")
    print("===================================")
    
    # Check if video2 is a duplicate
    is_duplicate, matches = detector.check_duplicate(video2_path, return_matches=True)
    
    # Print comparison details
    video1_id = os.path.basename(video1_path)
    video2_id = "query_" + os.path.basename(video2_path)  # Temporary ID used during check_duplicate
    
    # Get video data
    video1_data = detector.video_database[video1_id]
    
    # Calculate individual similarity metrics
    if is_duplicate and matches:
        print(f"\nResults: Video is a DUPLICATE")
        print(f"Overall similarity threshold: {detector.threshold:.2f}")
        print("\nMatches found:")
        for match_id, similarity in matches:
            print(f"  Match: {match_id}, Overall Similarity: {similarity:.4f}")
            
            # Calculate individual similarity metrics
            match_data = detector.video_database[match_id]
            
            frame_similarity = detector._compare_frame_hashes(
                video1_data['frame_hashes'], 
                match_data['frame_hashes']
            )
            
            audio_similarity = detector._compare_audio_features(
                video1_data['audio_features'], 
                match_data['audio_features']
            )
            
            motion_similarity = detector._compare_motion_signatures(
                video1_data['motion_signature'], 
                match_data['motion_signature']
            )
            
            color_similarity = detector._compare_color_histograms(
                video1_data['color_histogram'],
                match_data['color_histogram']
            )
            
            scene_similarity = detector._compare_scene_structure(
                video1_data['scene_changes'],
                match_data['scene_changes'],
                video1_data['frame_count'],
                match_data['frame_count']
            )
            
            # Print metadata comparison
            print("\nDetailed Similarity Metrics:")
            print(f"  Frame Hash Similarity: {frame_similarity:.4f}")
            print(f"  Audio Similarity: {audio_similarity:.4f}")
            print(f"  Motion Similarity: {motion_similarity:.4f}")
            print(f"  Color Similarity: {color_similarity:.4f}")
            print(f"  Scene Structure Similarity: {scene_similarity:.4f}")
            
            # Print video metadata
            v1_meta = video1_data['metadata']
            v2_meta = match_data['metadata']
            
            print("\nVideo Metadata Comparison:")
            print(f"  Video 1: {video1_id}")
            print(f"    Resolution: {v1_meta['width']}x{v1_meta['height']}")
            print(f"    Duration: {v1_meta['duration']:.2f} seconds")
            print(f"    FPS: {v1_meta['fps']:.2f}")
            print(f"    Aspect Ratio: {v1_meta['aspect_ratio']:.4f}")
            print(f"    Frame Count: {video1_data['frame_count']}")
            
            print(f"\n  Video 2: {match_id}")
            print(f"    Resolution: {v2_meta['width']}x{v2_meta['height']}")
            print(f"    Duration: {v2_meta['duration']:.2f} seconds")
            print(f"    FPS: {v2_meta['fps']:.2f}")
            print(f"    Aspect Ratio: {v2_meta['aspect_ratio']:.4f}")
            print(f"    Frame Count: {match_data['frame_count']}")
            
    else:
        print("\nResults: Video is UNIQUE (No duplicates found)")
    
    # Visualize comparison
    if is_duplicate and matches:
        print("\nGenerating visual comparison...")
        detector.visualize_comparison(video1_path, video2_path)

# Add this line to actually call the function when the script is run
if __name__ == "__main__":
    example_usage()