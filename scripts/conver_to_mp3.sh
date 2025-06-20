#!/bin/bash
find . -name "*.wav" | while read audio;do
  ffmpeg -i "$audio" "${audio%.*}.mp3"
done

