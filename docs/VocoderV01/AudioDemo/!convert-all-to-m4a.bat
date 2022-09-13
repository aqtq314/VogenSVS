for /r %%f in (*.wav *.flac) do (
    ffmpeg -i "%%f" -y -c:a aac -b:a 160k -ar 44100 -ac 1 "%%~nf.m4a"
    @REM echo %%f -^> %%~nf.m4a
    @REM del "%%f"
)