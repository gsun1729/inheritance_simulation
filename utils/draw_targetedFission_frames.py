from lib.playback import PlaybackSimulation
if __name__ == "__main__":
    
    import sys
    
    
    p = PlaybackSimulation(sys.argv[-1])
    p.playthroughTargFis(depth=3)
    
    