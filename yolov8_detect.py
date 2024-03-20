import m3u8

playlist = m3u8.load('http://136.169.226.59/1-4/tracks-v1/mono.m3u8?token=6284b70b0d85495a936206085e854b40')  # this could also be an absolute filename
print(playlist.segments)
print(playlist.target_duration)

# if you already have the content as string, use

playlist = m3u8.loads('#EXTM3U8 ... etc ... ')