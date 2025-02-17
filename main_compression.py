from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from pathlib import Path
import ffmpeg
import logging
from typing import Optional
import time

app = FastAPI(title="Video Processing App")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Create upload and processed directories if they don't exist
UPLOAD_DIR = Path("uploads")
PROCESSED_DIR = Path("processed")
UPLOAD_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

# Set up logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add root route to redirect to index.html
@app.get("/")
@app.get("/converter")
@app.get("/compressor")
async def serve_app():
    """Serve the main application for all main routes"""
    return FileResponse("static/index.html")

def get_video_duration(file_path):
    try:
        probe = ffmpeg.probe(str(file_path))
        duration = float(probe['format']['duration'])
        return round(duration, 2)
    except Exception as e:
        logger.error(f"Error getting video duration: {str(e)}")
        return None

@app.post("/convert")
async def convert_video(
    video: UploadFile = File(...),
    target_format: str = Form(...)
):
    logger.info(f"Starting conversion process for {video.filename} to {target_format}")
    start_time = time.time()
    
    try:
        # Save uploaded file
        video_path = UPLOAD_DIR / video.filename
        with open(video_path, "wb") as buffer:
            content = await video.read()
            buffer.write(content)
        logger.info(f"File saved successfully to {video_path}")
        
        # Get video duration
        duration = get_video_duration(video_path)
        
        # Generate output filename
        output_filename = f"{video_path.stem}.{target_format}"
        output_path = PROCESSED_DIR / output_filename
        logger.info(f"Output will be saved to {output_path}")

        try:
            # Set up FFmpeg command
            stream = (ffmpeg.input(str(video_path)).output(str(output_path),acodec='aac' if target_format == 'mp4' else 'copy',
                       vcodec='libx264' if target_format in ['mp4', 'avi'] else 'copy',
                       strict='experimental')
            )

            # Run the conversion
            logger.info("Starting FFmpeg conversion...")
            ffmpeg.run(stream, capture_stderr=True, overwrite_output=True)
            logger.info("FFmpeg conversion completed")

            # Verify the output file
            if not output_path.exists():
                raise Exception("Output file was not created")
            
            if output_path.stat().st_size == 0:
                raise Exception("Output file is empty")

            logger.info(f"Output file size: {output_path.stat().st_size} bytes")
            
            # Clean up input file
            os.remove(video_path)
            logger.info("Cleaned up input file")
            
            # Calculate elapsed time
            elapsed_time = round(time.time() - start_time, 2)
            
            return {
                "filename": output_filename,
                "duration": duration,
                "conversion_time": elapsed_time
            }

        except ffmpeg.Error as e:
            error_message = e.stderr.decode() if e.stderr else str(e)
            logger.error(f"FFmpeg error: {error_message}")
            return JSONResponse(
                status_code=500,
                content={"error": f"Conversion error: {error_message}"}
            )

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": f"Processing error: {str(e)}"}
        )

@app.post("/compress")
async def compress_video(
    video: UploadFile = File(...),
    compression_rate: float = Form(...)
):
    logger.info(f"Starting compression process for {video.filename}")
    start_time = time.time()
    
    try:
        # Save uploaded file
        video_path = UPLOAD_DIR / video.filename
        with open(video_path, "wb") as buffer:
            content = await video.read()
            buffer.write(content)
        logger.info(f"File saved successfully to {video_path}")
        
        # Generate output filename
        output_filename = f"{video_path.stem}_compressed.mp4"
        output_path = PROCESSED_DIR / output_filename
        
        try:
            # Get video information
            probe = ffmpeg.probe(str(video_path))
            logger.info(f"Probe result: {probe}")
            
            # Set default compression settings based on compression_rate
            crf_value = int(18 + (compression_rate / 100.0) * 10)
            
            # Build FFmpeg command with proper parameters
            stream = (
                ffmpeg
                .input(str(video_path))
                .output(
                    str(output_path),
                    acodec='aac',
                    vcodec='libx264',
                    preset='medium',
                    crf=str(crf_value)
                )
            )
            
            # Run the compression
            logger.info(f"Starting FFmpeg compression with CRF {crf_value}...")
            ffmpeg.run(stream, capture_stderr=True, overwrite_output=True)
            logger.info("Compression completed")
            
            # Verify the output file
            if not output_path.exists():
                raise Exception("Output file was not created")
            
            if output_path.stat().st_size == 0:
                raise Exception("Output file is empty")
                
            # Get file sizes for comparison
            original_size = os.path.getsize(str(video_path))
            compressed_size = os.path.getsize(str(output_path))
            compression_ratio = ((original_size - compressed_size) / original_size) * 100
            
            # Clean up input file
            os.remove(video_path)
            
            # Calculate elapsed time
            elapsed_time = round(time.time() - start_time, 2)
            
            return {
                "filename": output_filename,
                "original_size": f"{original_size / (1024*1024):.2f} MB",
                "compressed_size": f"{compressed_size / (1024*1024):.2f} MB",
                "compression_ratio": f"{compression_ratio:.1f}%",
                "compression_time": elapsed_time
            }
            
        except ffmpeg.Error as e:
            error_message = e.stderr.decode() if e.stderr else str(e)
            logger.error(f"FFmpeg error: {error_message}")
            if os.path.exists(str(video_path)):
                os.remove(video_path)
            return JSONResponse(
                status_code=500,
                content={"error": f"Compression error: {error_message}"}
            )
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        if os.path.exists(str(video_path)):
            os.remove(video_path)
        return JSONResponse(
            status_code=500,
            content={"error": f"Processing error: {str(e)}"}
        )

@app.get("/download/{filename}")
async def download_video(filename: str):
    logger.info(f"Download requested for {filename}")
    file_path = PROCESSED_DIR / filename
    
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return JSONResponse(
            status_code=404,
            content={"error": "File not found"}
        )
    
    logger.info(f"Serving file: {file_path}")
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="video/mp4"
    ) 