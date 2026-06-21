import cv2
import numpy as np
import json 
import argparse
import os

# prisha note = defines zones interactively and saves to a json file

# prisha note = selecting pretty colors :)
ZONE_COLORS = [
    (255, 100, 100), 
    (100, 255, 100), 
    (100, 100, 255), 
    (255, 255, 100), 
    (255, 100, 255), 
    (100, 255, 255)
]

def get_first_frame(video_path): 
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): 
        raise FileNotFoundError(f"Could not open video: {video_path}")
    ret, frame = cap.read()
    cap.release()
    if not ret: 
        raise RuntimeError("Could not read frame from video")
    return frame


def draw_state(base_img, zones, current_zone, current_index, zone_names): 
    display = base_img.copy()

    # prisha note = draw all zones
    for i, (name, pts) in enumerate(zones.items()):
        color = ZONE_COLORS[i % len(ZONE_COLORS)]
        arr = np.array(pts, dtype=np.int32)
        overlay = display.copy()
        cv2.fillPoly(overlay, [arr], color)
        cv2.addWeighted(overlay, .25, display, .75, display)
        cv2.polylines(display, [arr], True, color, 2)
        cx = int(np.mean([p[0] for p in pts]))
        cy = int(np.mean([p[1] for p in pts]))
        cv2.putText(display, name, (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        color = ZONE_COLORS[current_index % len(ZONE_COLORS)]
        for pt in current_zone: 
            cv2.circle(display, pt, 5, color, -1)
        if len(current_zone) >= 2: 
            for i in range(len(current_zone) - 1): 
                cv2.line(display, current_zone[i], current_zone[i + 1], color, 2)
            if len(current_zone) >= 3:
                cv2.line(display, current_zone[-1], current_zone[0], color, 1)
        if current_index < len(zone_names): 
            zone_name = zone_names[current_index]
        else:
            zone_name = f"Zone {current_index + 1}"
            instructions = [f"Drawing: '{zone_name}' (zone {current_index + 1})", "Left click = add point | N = finish zone | U = unbdo last point | Q = quit and save", f"Points so far: {len(current_zone)} (need at least 3)"]

        for i, text in enumerate(instructions): 
            cv2.putText(display, text, (10, 25 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return display
        
def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True, help="Path to video file")
    args = parser.parse_args()
    video_path = args.video
    frame = get_first_frame(video_path)

    ## prisha note = asks how many zones!
    print("\n=== Zone Definition Tool ===")
    print(f"Video: {video_path}")
    num_zones = int(input("How many zones do you want to define? "))
    zone_names = []
    for i in range(num_zones):
        name = input(f" Name for zone {i + 1} (examples include 'lane_left', 'turning_left'):").strip()
        zone_names.append(name if name else f"Zone {i + 1}")
        print ("\nWindow opened. Click to define each zone polygon")
        print("N = finish current zone | U = undo last point | Q = quit and save\n")

        zones = {}
        current_zone = []
        current_index = [0]

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN: 
                current_zone.append((x, y))
                cv2.imshow("Define Zones", draw_state(frame, zones, current_zone, current_index[0], zone_names))
            cv2.imshow("Define Zones", draw_state(frame, zones, current_zone, current_index[0], zone_names))
            cv2.setMouseCallback("Define Zones", mouse_callback)

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('n'):
                if len(current_zone) < 3: 
                    print("Need at least 3 points to define a zone")
                    continue
                    zone_name = zone_names[current_index[0]] if current_index[0] < len(zone_names) else f"zone_{current_index[0]+1}"
                    zones[zone_name] = current_zone.copy()
                    print(f"Saved Zone '{zone_name}' defined with {len(current_zone)} points")
                    current_zone.clear()
                    current_index[0] += 1
                    cv2.imshow("Define Zones", draw_state(frame, zones, current_zone, current_index[0], zone_names))
                    if current_index[0] >= num_zones:
                        print("All zones defined!! Yayy!")
                        break

                elif key == ord('u'):
                    if current_zone: 
                        current_zone.pop()
                        cv2.imshow("Define Zones", draw_state(frame, zones, current_zone, current_index[0], zone_names))

                elif key == ord('q'):
                    if len(current_zone) >= 3 and current_index[0] < len(zone_names):
                        zone_name = zone_names[current_index[0]]
                        zones[zone_name] = current_zone.copy()
                        print(f"Saved Zone '{zone_name}' ")
                    break

        cv2.destroyAllWindows()


        if not zones: 
            print("No zones defined, exiting without saving.")
            return

        ## prisha note: saving JSON next to video
        video_dir = os.path.dirname(video_path)
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        config = {
            "video": video_path,
            "zones": zones
        }

        with open(output_path, 'w') as f: 
            json.dump(config, f, indent=2)

        print(f"\nSaved zones config to {output_path}")
        print(json.dumps(config, indent=2))

if __name__ == "__main__":
    main()

