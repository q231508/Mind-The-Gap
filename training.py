
import os
import random
import time
 
import cv2
import mss
import numpy as np
import pygetwindow as gw
import pydirectinput
from scipy.special import softmax
import pickle

 
WINDOW_TITLE = "Celeste"

def find_celeste():
    windows = gw.getWindowsWithTitle(WINDOW_TITLE)
 
    if not windows:
        raise RuntimeError("Celeste not running :/")
    if windows[0].isMinimized:
        windows[0].restore()
        time.sleep(0.5)
        
    try:
        size = open("size.txt", "r")
        dims = size.readlines()
        windows[0].resizeTo(int(dims[0]), int(dims[1]))
    except:
        size = open("size.txt", "w")
        size.write(f"{str(windows[0].width)}\n")
        size.write(f"{str(windows[0].height)}\n")
    size.close()
    windows[0].moveTo(0, 0)
    
    return windows[0]
 
 
def rgb_to_hsv(rgb):
    pixel = np.uint8([[rgb]])
    return cv2.cvtColor(pixel, cv2.COLOR_RGB2HSV)[0, 0]
 
 
def build_mask(hsv, lower, upper):
    lower_red_1 = np.array([170, 90, 100], dtype=np.uint8)
    upper_red_1 = np.array([179, 255, 255], dtype=np.uint8)
 
    lower_red_2 = np.array([0, 90, 100], dtype=np.uint8)
    upper_red_2 = np.array([5, 255, 255], dtype=np.uint8)
 
    lower_dash = np.array(lower[0], dtype=np.uint8)
    upper_dash = np.array(upper[0], dtype=np.uint8)

    lower_skin = np.array(lower[1], dtype=np.uint8)
    upper_skin = np.array(upper[1], dtype=np.uint8)
    
    lower_bag = np.array(lower[2], dtype=np.uint8)
    upper_bag = np.array(upper[2], dtype=np.uint8)
    
    red_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    red_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    dash = cv2.inRange(hsv, lower_dash, upper_dash)
    skin = cv2.inRange(hsv, lower_skin, upper_skin)
    bag = cv2.inRange(hsv, lower_bag, upper_bag)
    

    mask = cv2.bitwise_or(red_1, red_2)
    mask = cv2.bitwise_or(mask, dash)
    mask = cv2.bitwise_or(mask, skin)
    mask = cv2.bitwise_or(mask, bag)

    return mask
 
def is_frame_black(frame, threshold):
    mean_val = cv2.mean(frame)  
    return max(mean_val[:3]) < threshold

def mouse_callback(event, x, y, flags, param):
    global drawing, start_point, current_point, display, frame, calibrated, initial_x, initial_y, initial_w, initial_h

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        current_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        current_point = (x, y)

        display = frame.copy()
        cv2.rectangle(
            display,
            start_point,
            current_point,
            (0, 255, 0),
            2
        )

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        current_point = (x, y)

        cv2.rectangle(
            display,
            start_point,
            current_point,
            (0, 255, 0),
            2
        )
        
        cv2.circle(
            display,
            (start_point[0] + ((current_point[0] - start_point[0]) // 2), start_point[1] + ((current_point[1] - start_point[1]) // 2)),
            4,
            (0, 255, 0),
            -1
        )

        #print("Selected ROI:", start_point, current_point)
        
        
        calibrated = True
        initial_x = start_point[0]
        initial_y = start_point[1]
        initial_w = abs(current_point[0] - start_point[0])
        initial_h = abs(current_point[1] - start_point[1])




def main():
    try:
        rgb_madeline_pixels = [(75, 244, 255), (243, 218, 179), (141, 67, 53)]
        
        HSVs = []
        for color in rgb_madeline_pixels:
            HSVs.append(rgb_to_hsv(color))
        
        lower = []
        upper = []
        for hsv_val in HSVs:
            l = np.array([
                max(0, int(hsv_val[0]) - 10),
                max(0, int(hsv_val[1]) - 5),
                max(0, int(hsv_val[2]) - 50)
            ], dtype=np.uint8)
            lower.append(l)
        
            u = np.array([
                min(179, int(hsv_val[0]) + 10),
                min(255, int(hsv_val[1]) + 30),
                min(255, int(hsv_val[2]) + 20)
            ], dtype=np.uint8)
            upper.append(u)
        
        #print(lower)
        #print(upper)
        
        # Finds the celeste window
        celeste = find_celeste()
        

        
        celeste.activate()
        time.sleep(1)
        
        # Restarts the level
        pydirectinput.press("esc")
        pydirectinput.press("up")
        pydirectinput.press("up")
        pydirectinput.press("c")
        pydirectinput.press("c")
        pydirectinput.PAUSE = 0.01
        
        pydirectinput.keyUp("up")
        pydirectinput.keyUp("down")
        pydirectinput.keyUp("left")
        pydirectinput.keyUp("right")
        pydirectinput.keyUp("c")
        pydirectinput.keyUp("z")
        time.sleep(4)

        # Create the OpenCV display window
        cv2.namedWindow("compvis capture", cv2.WINDOW_NORMAL)
        
        # Termination critea for the tracking window
        term_crit = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            10,
            1,
        )
        
        # HSV filter to remove extremely dark/bright pixels
        SV_FLOOR_LOWER = (0, 60, 60)
        SV_FLOOR_UPPER = (179, 255, 255)
        
        actions = [
                "holdup", 
                "holdright", 
                "holdz", 
                "holdc",
                "releaseup", 
                "releasez"]
        
        last_x = 0
        last_y = 0
        last_action = None
        r = None
        alpha = 0.5
        gamma = 1
        took_action = False
        cleared = False
        clears = 0
        deaths = 0
        attempts = 0
        
        MOVEMENT = -.1
        DYING = -10
        ZONE1 = 2
        ZONE2 = 3
        GOAL = 100

        random.seed(0)
        
        global drawing, display, calibrated
        drawing = False

        calibrated = False
        visited_reward_zone_1_this_episode = False
        visited_reward_zone_2_this_episode = False
        jumped = False
        t = time.time()
        file = open("stats.txt", "w")
        distance = 0
        loaded = False
        with mss.mss() as screenshotter:
            try:
                # Attempts to load in a previous policy
                if os.path.isfile("losl23.pkl"):
                    with open("losl23.pkl", "rb") as f:
                        states = pickle.load(f)
                        loaded = True
                else:
                    states = {}
                dead = False
                change_time = time.time()
                lj = False
            

                while True:
                    if (time.time() - t) > 1/60:
                        t = time.time()
                        global frame, initial_x, initial_h, initial_w, initial_y
                        
                        # Get the current Celeste window position and size
                        left = celeste.left 
                        top = celeste.top
                        width = celeste.width
                        height = celeste.height
            
                        if width <= 0 or height <= 0:
                            time.sleep(0.1)
                            continue
                        
                        
                            
            
                        monitor = {
                            "left": left,
                            "top": top,
                            "width": width,
                            "height": height,
                        }
            
                        # Capture the Celeste window
                        screenshot = screenshotter.grab(monitor)
            
                        # Convert screenshot image to an OpenCV BGR image
                        frame = np.array(screenshot)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
                        # Convert the full frame to HSV
                        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                        
                        if not calibrated:
                            display = frame
                            try:
                                f2 = open("calibration.txt", "r")
                                cal = f2.readlines()
                                initial_x = int(cal[0])
                                initial_y = int(cal[1])
                                initial_w = int(cal[2])
                                initial_h = int(cal[3])
                                track_window = (
                                    initial_x,
                                    initial_y,
                                    initial_w,
                                    initial_h,
                                )
                            except:
                                

                                cv2.namedWindow("calibration")
                                cv2.setMouseCallback("calibration", mouse_callback)
                                #print("lol")
                                while not calibrated:
                                    cv2.imshow("calibration", display)
                                    key = cv2.waitKey(1) & 0xFF

                                
                                cv2.destroyWindow("calibration")
                                track_window = (
                                    initial_x,
                                    initial_y,
                                    initial_w,
                                    initial_h,
                                )
                                
                                f2 = open("calibration.txt", "w")
                                f2.write(f"{str(initial_x)}\n")
                                f2.write(f"{str(initial_y)}\n")
                                f2.write(f"{str(initial_w)}\n")
                                f2.write(f"{str(initial_h)}\n")

                            if not loaded:
                                for x in range(width):
                                    for y in range(height):
                                        states[f"X:{x} Y:{y}"] = [0] * len(actions)
                            visits = np.zeros((width, height), dtype=np.uint32)
                            time.sleep(1)
                            last_x = initial_x + initial_w // 2
                            last_y = initial_y + initial_h // 2
                            last_s = f"X:{last_x} Y:{last_y}"
                            calibrated = True
                            f2.close


                        # color mask for Madeline's pixels
                        live_mask = build_mask(hsv, lower, upper)
            
                        sv_floor = cv2.inRange(hsv, SV_FLOOR_LOWER, SV_FLOOR_UPPER)
                        projection = cv2.bitwise_and(live_mask, sv_floor)
            
                        # Move the tracking window using MeanShift
                        _, track_window = cv2.meanShift(
                            projection,
                            track_window,
                            term_crit
                        )
                        
                        
                        
                        x, y, w, h = track_window

                        rgb = cv2.mean(frame[y:y+h, x:x+w])
                        mc = np.mean(rgb[:3])
                                                
                        if is_frame_black(frame, 20):
                            
                            attempts += 1
                            if not dead and not cleared:
                                print(f"you died lol :P Attempts: {attempts} Clears: {clears}")
                                
                                track_window = (
                                    initial_x,
                                    initial_y,
                                    initial_w,
                                    initial_h,
                                )
                                x, y, w, h = track_window
                                dead = True
                                deaths+=1

                                pydirectinput.keyUp("up")
                                pydirectinput.keyUp("down")
                                pydirectinput.keyUp("left")
                                pydirectinput.keyUp("right")
                                pydirectinput.keyUp("c")
                                pydirectinput.keyUp("z")
                                r = MOVEMENT
                                visited_reward_zone_1_this_episode = False
                                visited_reward_zone_2_this_episode = False
                                jumped = False
                                lj = False

                                time.sleep(2)

                            if cleared:
                                dead = False
                                cleared = False
                                print(f"Wow... You actually did it! Attempts: {attempts} Clears: {clears}")
                                
                                track_window = (
                                    initial_x,
                                    initial_y,
                                    initial_w,
                                    initial_h,
                                )
                                x, y, w, h = track_window
                                time.sleep(3)
                                r = DYING
                                took_action = False
                                pydirectinput.keyUp("up")
                                pydirectinput.keyUp("down")
                                pydirectinput.keyUp("left")
                                pydirectinput.keyUp("right")
                                pydirectinput.keyUp("c")
                                pydirectinput.keyUp("z")
                            change_time = time.time()

                            file.write(f"Deaths: {str(deaths)}, Clears: {str(clears)}, Attempts: {str(attempts)}, Distance: {str(distance)}\n")

                            #exit()
                        elif mc > 120:
                            attempts += 1

                            if not dead and not cleared:
                                print(f"you died lol :P Attempts: {attempts} Clears: {clears}")
                                
                                track_window = (
                                    initial_x,
                                    initial_y,
                                    initial_w,
                                    initial_h,
                                )
                                x, y, w, h = track_window
                                dead = True
                                deaths+=1
                                pydirectinput.keyUp("up")
                                pydirectinput.keyUp("down")
                                pydirectinput.keyUp("left")
                                pydirectinput.keyUp("right")
                                pydirectinput.keyUp("c")
                                pydirectinput.keyUp("z")
                                r = MOVEMENT
                                visited_reward_zone_1_this_episode = False
                                visited_reward_zone_2_this_episode = False
                                jumped = False
                                lj = False
                                time.sleep(2)
                            change_time = time.time()
                            file.write(f"Deaths: {str(deaths)}, Clears: {str(clears)}, Attempts: {str(attempts)}, Distance: {str(distance)}\n")

                        else:
                            dead = False
                            x, y, w, h = track_window
                            
                        # Draw the tracking rectangle
                        cv2.rectangle(
                            frame,
                            (x, y),
                            (x + w, y + h),
                            (0, 255, 0),  # Green 
                            2
                        )
            
                        # Draw the center point
                        center_x = x + w // 2
                        center_y = y + h // 2
            
                        cv2.circle(
                            frame,
                            (center_x, center_y),
                            4,
                            (255, 0, 0),  # Blue 
                            -1
                        )
            
                        cv2.rectangle(
                            projection,
                            (x, y),
                            (x + w, y + h),
                            (255, 0, 0),
                            2
                        )
            
                        cv2.circle(
                            projection,
                            (center_x, center_y),
                            4,
                            (255, 0, 0),
                            -1
                        )
            
                        match_strength = cv2.sumElems(projection)[0]
                        ##print(match_strength, track_window)
            
                        #cv2.imshow("projection", projection)
                        #cv2.imshow("compvis capture", frame)
                        change = False
                        distance = (center_x, center_y)
                        
                        if abs(center_x - last_x) >= 1 or abs(center_y- last_y) >= 1:
                            change = True
                            change_time = time.time()
                        else:
                            if time.time() - change_time >= 3:
                                dead = True
                                pydirectinput.keyUp("up")
                                pydirectinput.keyUp("down")
                                pydirectinput.keyUp("left")
                                pydirectinput.keyUp("right")
                                pydirectinput.keyUp("c")
                                pydirectinput.keyUp("z")
                                time.sleep(1)

                                
                                pydirectinput.press("esc")
                                pydirectinput.press("up")
                                pydirectinput.press("up")
                                pydirectinput.press("c")
                                pydirectinput.keyDown("c")
                        
                    
                        action_dist = softmax(np.array(states[f"X:{center_x} Y:{center_y}"]))
                        
                        if random.random() < 0.05:
                            a = random.choices(
                                actions,
                                weights=action_dist,
                                k=1
                            )
                            while a == ["releasez"] and lj:
                                a = random.choices(
                                    actions,
                                    weights=action_dist,
                                    k=1
                                )
                        else:
                            best = max(action_dist)
                            indices = [i for i, value in enumerate(action_dist) if value == best]
                            index = random.choice(indices)
                            a = [actions[index]]

    #                     if len(indices) < 6:
                                #print(indices)


                        # #print(a)
                                
                        if change or not change:
                            current_a = actions.index(a[0])
                            
                        # if not took_action:
                                #print(f"X: {center_x}px, Y: {center_y}px, Action: {a}")

                            if took_action:
                                if dead:
                                    r = DYING
                                    dead = False
                                elif center_x >= 740 and center_y < 220:
                                    r = GOAL
                                    pydirectinput.keyUp("up")
                                    pydirectinput.keyUp("down")
                                    pydirectinput.keyUp("left")
                                    pydirectinput.keyUp("right")
                                    pydirectinput.keyUp("c")
                                    pydirectinput.keyUp("z")
                                    time.sleep(1)
                                    
                                    if not cleared:
                                        
                                        cleared = True 
                                        clears += 1
                                    pydirectinput.PAUSE = 0.1

                                    pydirectinput.press("r")
                                    pydirectinput.press("c")
                                    pydirectinput.press("c")
                                    pydirectinput.keyDown("c")
                                    pydirectinput.PAUSE = 0.01

                                    
                                elif (240 < center_x and center_x < 310) and (410 >= center_y and center_y >= 310) and not visited_reward_zone_1_this_episode:
                                    r = ZONE1
                                    visited_reward_zone_1_this_episode = True
                                    #print(F"REWARD {ZONE1}")
                                    current_a = actions.index("holdz")
                                    a = ["releasez"]
                                elif (initial_x+w <= center_x and center_x <= initial_x+w*2) and (470 >= center_y and center_y >= 310) and actions[last_action] == "holdz" and not jumped:
                                    jumped = True
                                    r = 5
                                    #print(F"REWARD 5")
                                elif (250 <= center_x and center_x <= 320) and (410 >= center_y and center_y >= 410-h-20) and actions[last_action] == "holdz" and not jumped:
                                    jumped = True
                                    r = 5
                                    #print(F"REWARD 5 ZONE1 JUMP", center_x)
                                elif (530 <= center_x and center_x <= 550) and (270 >= center_y and center_y >= 270-h) and actions[last_action] == "holdz" and not jumped:
                                    jumped = True
                                    r = 10
                                    #print(F"REWARD 10 ZONE2 JUMP", center_x)
                                    lj = True
                                elif (430 <= center_x and center_x <= 530) and (270 >= center_y and center_y >= 180) and actions[last_action] == "holdz":
                                    jumped = True
                                    r = -5
                                    #print(F"REWARD -5 BAD ZONE2 JUMP", center_x)
                                    lj = True
                                elif (430 <= center_x and center_x <= 510) and (270 >= center_y and center_y >= 180) and not visited_reward_zone_2_this_episode:
                                    r = ZONE2
                                    visited_reward_zone_2_this_episode = True
                                    #print(F"REWARD {ZONE2}")
                                    current_a = actions.index("holdz")
                                    a = ["releasez"]

                                elif  time.time() - change_time >= 1 and (center_x == last_s and center_y == last_y):
                                    r = -10
                                    #print("Stop Stalling")
                                    
                                elif (560 < center_x  and center_x < 720) and actions[last_action] == "releasez":
                                    r = -10
                                    #print("You released jump...")


                                elif (center_x < 740 and center_y > 70):
                                    r = MOVEMENT
                                    cleared = False
                                    jumped = False




                                states[last_s][last_action] = states[last_s][last_action]  + alpha * ( r + gamma * (states[f"X:{center_x} Y:{center_y}"][current_a]) - states[last_s][last_action]  )
                                took_action = False
                                
                                
                            #   #print(f"{last_s}, Action: {actions[last_action]}, Reward: {r}")
                                if dead:
                                    last_x = initial_x + w // 2
                                    last_y = initial_y + h // 2
                                    last_s = f"X:{last_x} Y:{last_y}"
                                    

                            took_action = True

                            if change:
                                
                                last_s = f"X:{center_x} Y:{center_y}"

                                
                                    
                                change = False
                                last_x = center_x
                                last_y = center_y
                            last_action = actions.index(a[0])
                            if "release" in a[0]:
                                pydirectinput.keyUp(a[0].removeprefix("release"))
                            elif "hold" in a[0]:
                                pydirectinput.keyDown(a[0].removeprefix("hold"))

                            
                        visits[center_x, center_y] += 1
                        
                        # q or Esc to quit
                        key = cv2.waitKey(1) & 0xFF
            
                        if key == ord("q") or key == 27 or attempts == 10001:
                            pydirectinput.keyUp("up")
                            pydirectinput.keyUp("down")
                            pydirectinput.keyUp("left")
                            pydirectinput.keyUp("right")
                            pydirectinput.keyUp("c")
                            pydirectinput.keyUp("z")
                            np.save("fin.npy", visits)


                            with open("fin.pkl", "wb") as f:
                                pickle.dump(states, f)
                            break
                        

                        
                    
        
            finally:
                cv2.destroyAllWindows()
    except:
        pydirectinput.keyUp("up")
        pydirectinput.keyUp("down")
        pydirectinput.keyUp("left")
        pydirectinput.keyUp("right")
        pydirectinput.keyUp("c")
        pydirectinput.keyUp("z")
        np.save("fin.npy", visits)


        with open("fin.pkl", "wb") as f:
            pickle.dump(states, f)
        cv2.destroyAllWindows()
if __name__ == "__main__":
    main()