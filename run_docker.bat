@echo off
echo Building ZoeDepth Docker image...
docker build -t zoedepth:latest .

echo.
echo Running ZoeDepth container...
docker run -it --rm ^
    -v "%cd%":/app/zoedepth ^
    -v "%cd%\images":/app/images ^
    -v "%cd%\output":/app/output ^
    -p 8080:8080 ^
    zoedepth:latest

pause