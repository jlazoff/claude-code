#!/usr/bin/env python3
"""
YouTube CLI Integration
Uses youtube-dl/yt-dlp command line tools for more efficient video extraction
"""

import asyncio
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YouTubeCLIIntegration:
    def __init__(self):
        self.foundation_dir = Path("foundation_data")
        self.youtube_dir = self.foundation_dir / "youtube_data"
        self.youtube_dir.mkdir(exist_ok=True)
        
        # Check for yt-dlp installation
        self.ensure_youtube_tools()
        
        logger.info("YouTube CLI Integration initialized")

    def ensure_youtube_tools(self):
        """Ensure youtube-dl/yt-dlp and dependencies are installed"""
        tools_to_install = [
            ("yt-dlp", "pip3 install yt-dlp"),
            ("youtube-dl", "pip3 install youtube-dl"),
            ("ffmpeg", "brew install ffmpeg")
        ]
        
        for tool, install_cmd in tools_to_install:
            try:
                result = subprocess.run([tool, "--version"], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    logger.info(f"✅ {tool} is available")
                else:
                    logger.info(f"Installing {tool}...")
                    subprocess.run(install_cmd.split(), check=True, timeout=300)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
                logger.warning(f"⚠️ {tool} not available, installing...")
                try:
                    subprocess.run(install_cmd.split(), check=True, timeout=300)
                    logger.info(f"✅ {tool} installed successfully")
                except Exception as e:
                    logger.error(f"❌ Failed to install {tool}: {e}")

    async def extract_channel_metadata(self, channel_url: str) -> Dict[str, Any]:
        """Extract channel metadata using yt-dlp"""
        logger.info(f"📺 Extracting channel metadata: {channel_url}")
        
        cmd = [
            "yt-dlp",
            "--dump-json",
            "--flat-playlist",
            "--playlist-end", "100",  # Limit to first 100 videos
            channel_url
        ]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                logger.error(f"yt-dlp failed: {error_msg}")
                return {"error": error_msg, "videos": []}
            
            # Parse JSON output (one JSON object per line)
            videos = []
            for line in stdout.decode().strip().split('\n'):
                if line.strip():
                    try:
                        video_data = json.loads(line)
                        videos.append(video_data)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse JSON line: {e}")
            
            logger.info(f"✅ Extracted metadata for {len(videos)} videos")
            
            # Extract channel info from first video
            channel_info = {}
            if videos:
                first_video = videos[0]
                channel_info = {
                    "channel_id": first_video.get("channel_id"),
                    "channel": first_video.get("channel"),
                    "channel_url": first_video.get("channel_url"),
                    "uploader": first_video.get("uploader"),
                    "uploader_id": first_video.get("uploader_id")
                }
            
            return {
                "channel_info": channel_info,
                "video_count": len(videos),
                "videos": videos
            }
            
        except Exception as e:
            logger.error(f"Failed to extract channel metadata: {e}")
            return {"error": str(e), "videos": []}

    async def extract_video_details(self, video_id: str) -> Dict[str, Any]:
        """Extract detailed video information"""
        logger.info(f"🎬 Extracting video details: {video_id}")
        
        cmd = [
            "yt-dlp",
            "--dump-json",
            "--write-sub",
            "--write-auto-sub",
            "--sub-lang", "en",
            "--skip-download",
            f"https://www.youtube.com/watch?v={video_id}"
        ]
        
        try:
            # Use temporary directory for subtitle files
            with tempfile.TemporaryDirectory() as temp_dir:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=temp_dir
                )
                
                stdout, stderr = await process.communicate()
                
                if process.returncode != 0:
                    error_msg = stderr.decode() if stderr else "Unknown error"
                    logger.warning(f"yt-dlp failed for {video_id}: {error_msg}")
                    return {"error": error_msg}
                
                # Parse video metadata
                video_data = json.loads(stdout.decode())
                
                # Extract transcript from subtitle files
                transcript = ""
                temp_path = Path(temp_dir)
                
                # Look for subtitle files
                for sub_file in temp_path.glob(f"{video_id}*.vtt"):
                    try:
                        transcript_text = self.parse_vtt_subtitle(sub_file)
                        if transcript_text:
                            transcript = transcript_text
                            break
                    except Exception as e:
                        logger.warning(f"Failed to parse subtitle file {sub_file}: {e}")
                
                # If no VTT files, look for other formats
                if not transcript:
                    for sub_file in temp_path.glob(f"{video_id}*"):
                        if sub_file.suffix in ['.srt', '.ass', '.ttml']:
                            try:
                                transcript_text = sub_file.read_text(encoding='utf-8')
                                if transcript_text:
                                    transcript = self.clean_subtitle_text(transcript_text)
                                    break
                            except Exception as e:
                                logger.warning(f"Failed to read subtitle file {sub_file}: {e}")
                
                result = {
                    "video_id": video_id,
                    "title": video_data.get("title", ""),
                    "description": video_data.get("description", ""),
                    "duration": video_data.get("duration", 0),
                    "view_count": video_data.get("view_count", 0),
                    "like_count": video_data.get("like_count", 0),
                    "upload_date": video_data.get("upload_date", ""),
                    "uploader": video_data.get("uploader", ""),
                    "tags": video_data.get("tags", []),
                    "categories": video_data.get("categories", []),
                    "transcript": transcript,
                    "automatic_captions": bool(video_data.get("automatic_captions")),
                    "subtitles": bool(video_data.get("subtitles"))
                }
                
                logger.info(f"✅ Extracted details for video: {result['title'][:50]}...")
                return result
                
        except Exception as e:
            logger.error(f"Failed to extract video details for {video_id}: {e}")
            return {"error": str(e)}

    def parse_vtt_subtitle(self, vtt_file: Path) -> str:
        """Parse VTT subtitle file to extract text"""
        try:
            content = vtt_file.read_text(encoding='utf-8')
            lines = content.split('\n')
            
            transcript_lines = []
            skip_next = False
            
            for line in lines:
                line = line.strip()
                
                # Skip VTT headers and timing lines
                if line.startswith('WEBVTT') or line.startswith('NOTE') or '-->' in line:
                    continue
                
                # Skip empty lines and timestamp lines
                if not line or line.isdigit():
                    continue
                
                # Skip lines that look like timestamps
                if ':' in line and len(line.split(':')) >= 2:
                    try:
                        # Check if it's a timestamp format
                        parts = line.split(' --> ')[0] if ' --> ' in line else line
                        time_parts = parts.split(':')
                        if len(time_parts) >= 2:
                            float(time_parts[-1])  # Try to parse seconds
                            continue
                    except (ValueError, IndexError):
                        pass
                
                # Clean and add text line
                clean_line = self.clean_subtitle_line(line)
                if clean_line and clean_line not in transcript_lines:
                    transcript_lines.append(clean_line)
            
            return ' '.join(transcript_lines)
            
        except Exception as e:
            logger.warning(f"Failed to parse VTT file {vtt_file}: {e}")
            return ""

    def clean_subtitle_line(self, line: str) -> str:
        """Clean subtitle line of formatting tags"""
        # Remove common subtitle formatting
        import re
        
        # Remove HTML-like tags
        line = re.sub(r'<[^>]+>', '', line)
        
        # Remove VTT formatting
        line = re.sub(r'\{[^}]+\}', '', line)
        
        # Remove speaker labels like [Music] or (applause)
        line = re.sub(r'\[[^\]]+\]', '', line)
        line = re.sub(r'\([^)]+\)', '', line)
        
        # Clean up whitespace
        line = ' '.join(line.split())
        
        return line.strip()

    def clean_subtitle_text(self, text: str) -> str:
        """Clean subtitle text content"""
        lines = text.split('\n')
        clean_lines = []
        
        for line in lines:
            clean_line = self.clean_subtitle_line(line)
            if clean_line and not clean_line.isdigit() and '-->' not in clean_line:
                clean_lines.append(clean_line)
        
        return ' '.join(clean_lines)

    async def analyze_channel(self, channel_url: str) -> Dict[str, Any]:
        """Comprehensive channel analysis using CLI tools"""
        logger.info(f"🔍 Starting comprehensive analysis: {channel_url}")
        
        # Extract channel metadata
        channel_data = await self.extract_channel_metadata(channel_url)
        
        if "error" in channel_data:
            return channel_data
        
        videos = channel_data["videos"]
        logger.info(f"📊 Analyzing {len(videos)} videos...")
        
        # Analyze first 10 videos in detail
        detailed_videos = []
        for i, video in enumerate(videos[:10]):
            video_id = video.get("id")
            if video_id:
                logger.info(f"Processing video {i+1}/10: {video.get('title', 'Unknown')[:50]}...")
                details = await self.extract_video_details(video_id)
                if "error" not in details:
                    detailed_videos.append(details)
                await asyncio.sleep(1)  # Rate limiting
        
        # Compile analysis results
        analysis_result = {
            "channel_info": channel_data["channel_info"],
            "total_videos": channel_data["video_count"],
            "analyzed_videos": len(detailed_videos),
            "videos": detailed_videos,
            "analysis_summary": {
                "total_duration": sum(v.get("duration", 0) for v in detailed_videos),
                "total_views": sum(v.get("view_count", 0) for v in detailed_videos),
                "total_likes": sum(v.get("like_count", 0) for v in detailed_videos),
                "videos_with_transcripts": len([v for v in detailed_videos if v.get("transcript")]),
                "common_tags": self.extract_common_tags(detailed_videos),
                "upload_frequency": self.analyze_upload_frequency(detailed_videos)
            }
        }
        
        # Save analysis to foundation
        channel_name = channel_data["channel_info"].get("channel", "unknown_channel")
        safe_name = "".join(c for c in channel_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        
        analysis_file = self.youtube_dir / f"analysis_{safe_name}_{int(asyncio.get_event_loop().time())}.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"✅ Analysis complete: {analysis_file}")
        return analysis_result

    def extract_common_tags(self, videos: List[Dict[str, Any]]) -> List[str]:
        """Extract most common tags across videos"""
        tag_counts = {}
        
        for video in videos:
            for tag in video.get("tags", []):
                tag_lower = tag.lower()
                tag_counts[tag_lower] = tag_counts.get(tag_lower, 0) + 1
        
        # Return top 10 most common tags
        sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
        return [tag for tag, count in sorted_tags[:10]]

    def analyze_upload_frequency(self, videos: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze upload frequency patterns"""
        if not videos:
            return {"frequency": "unknown", "pattern": "insufficient_data"}
        
        upload_dates = []
        for video in videos:
            upload_date = video.get("upload_date")
            if upload_date:
                try:
                    # Convert YYYYMMDD to datetime
                    year = int(upload_date[:4])
                    month = int(upload_date[4:6])
                    day = int(upload_date[6:8])
                    upload_dates.append((year, month, day))
                except (ValueError, IndexError):
                    continue
        
        if len(upload_dates) < 2:
            return {"frequency": "unknown", "pattern": "insufficient_data"}
        
        # Calculate average days between uploads
        upload_dates.sort()
        gaps = []
        
        for i in range(1, len(upload_dates)):
            prev_date = upload_dates[i-1]
            curr_date = upload_dates[i]
            
            # Simple day difference calculation
            prev_days = prev_date[0] * 365 + prev_date[1] * 30 + prev_date[2]
            curr_days = curr_date[0] * 365 + curr_date[1] * 30 + curr_date[2]
            
            gap = abs(curr_days - prev_days)
            if gap > 0:
                gaps.append(gap)
        
        if gaps:
            avg_gap = sum(gaps) / len(gaps)
            
            if avg_gap <= 3:
                frequency = "daily"
            elif avg_gap <= 10:
                frequency = "weekly"
            elif avg_gap <= 35:
                frequency = "monthly"
            else:
                frequency = "irregular"
            
            return {
                "frequency": frequency,
                "average_gap_days": avg_gap,
                "pattern": "consistent" if max(gaps) - min(gaps) < avg_gap else "variable"
            }
        
        return {"frequency": "unknown", "pattern": "insufficient_data"}

async def main():
    """Test the YouTube CLI integration"""
    cli = YouTubeCLIIntegration()
    
    # Test with @code4AI channel
    channel_url = "https://www.youtube.com/@code4AI"
    
    try:
        print(f"🔍 Analyzing channel: {channel_url}")
        result = await cli.analyze_channel(channel_url)
        
        if "error" in result:
            print(f"❌ Analysis failed: {result['error']}")
            return
        
        print(f"\n✅ Analysis complete!")
        print(f"   Channel: {result['channel_info'].get('channel', 'Unknown')}")
        print(f"   Total Videos: {result['total_videos']}")
        print(f"   Analyzed Videos: {result['analyzed_videos']}")
        print(f"   Videos with Transcripts: {result['analysis_summary']['videos_with_transcripts']}")
        print(f"   Total Duration: {result['analysis_summary']['total_duration']} seconds")
        print(f"   Upload Frequency: {result['analysis_summary']['upload_frequency']['frequency']}")
        
        if result['analysis_summary']['common_tags']:
            print(f"   Common Tags: {', '.join(result['analysis_summary']['common_tags'][:5])}")
        
        print(f"\n📹 Sample Videos:")
        for video in result['videos'][:3]:
            print(f"   • {video['title'][:60]}...")
            print(f"     Duration: {video['duration']}s, Views: {video['view_count']}")
            if video['transcript']:
                print(f"     Transcript: {len(video['transcript'])} characters")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())