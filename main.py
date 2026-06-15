"""
Main runner to load boat and sea modeling and start animation.
"""
from boat import create_boat_mesh
import jonswap


def main():
    # Ensure modules are imported and animation is started
    # create_boat_mesh is available from boat.py and used by jonswap
    print('Starting sea + boat animation...')
    jonswap.start_animation()


if __name__ == '__main__':
    main()
