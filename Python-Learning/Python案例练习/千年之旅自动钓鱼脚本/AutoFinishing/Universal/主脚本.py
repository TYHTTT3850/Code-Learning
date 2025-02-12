import pyautogui
import cv2
import numpy as np
import time
import sys

# 预定义图像模板路径
JOYSTICK_IMAGE = r'D:\AutoFinishing\Universal\joystick.png'  # 摇杆图标模板
FISH_ON_HOOK_IMAGE = r'D:\AutoFinishing\Universal\shou_gan.png'  # 收杆提示
END_IMAGE = r'D:\AutoFinishing\Universal\end.png' # 结算画面

# 加载图像模板
def load_image(template_path):
    return cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)


# 自动检测屏幕上图像的位置
def find_on_screen(template, confidence=0.5):
    screen = pyautogui.screenshot()
    screen_np = np.array(screen)
    screen_gray = cv2.cvtColor(screen_np, cv2.COLOR_BGR2GRAY)

    res = cv2.matchTemplate(screen_gray, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= confidence)

    # 如果找到匹配，则返回模板的中心坐标
    if len(loc[0]) > 0:
        top_left = (loc[1][0], loc[0][0])
        h, w = template.shape
        center_x = top_left[0] + w // 2
        center_y = top_left[1] + h // 2
        return center_x, center_y
    return

# 点击屏幕上的坐标
def click_on_screen(x, y):
    pyautogui.click(x, y)
    time.sleep(0.01)

# 自动检测摇杆位置
def detect_joystick_position():
    print("正在加载摇杆模板...")
    template = load_image(JOYSTICK_IMAGE)

    print("正在检测摇杆位置...")
    joystick_pos = find_on_screen(template, confidence=0.6)

    if joystick_pos:
        print(f"摇杆位置检测到: {joystick_pos}")
        return joystick_pos
    else:
        print("未检测到摇杆，请检查模板图片或调整匹配度")
        return None

# 模拟旋转摇杆
def rotate_joystick(joystick_pos):
    radius = 250  # 旋转半径
    center_x, center_y = joystick_pos
    for angle in range(0, 361, 45):
        radian = np.radians(angle)
        x = center_x + int(radius * np.cos(radian))
        y = center_y + int(radius * np.sin(radian))
        pyautogui.moveTo(x, y)
        time.sleep(0.001)


"""-----------------------------主程序-----------------------------------"""
joystick_pos = list(detect_joystick_position()) #检测摇杆位置
joystick_pos[0] += 600
joystick_pos[1] -= 200

click_pos = [0,0]
click_pos[0] =joystick_pos[0] + 100
click_pos[1] = joystick_pos[1] + 100

fish_on_hook_template = load_image(FISH_ON_HOOK_IMAGE) #加载收杆提示
end_template = load_image(END_IMAGE) #加载结算画面

# 遥感被遮挡，退出程序
if joystick_pos is None:
    print("无法检测到摇杆，程序退出")
    sys.exit()

while True:

    # 抛竿
    click_on_screen(joystick_pos[0],joystick_pos[1])
    condition1 = bool(find_on_screen(fish_on_hook_template, confidence=0.6))

    # 等待收杆条件成立
    while not condition1:
        condition1 = bool(find_on_screen(fish_on_hook_template, confidence=0.6))


    # 收杆
    click_on_screen(click_pos[0], click_pos[1])
    pyautogui.mouseDown(button='left')
    
    # 持续拖动摇杆
    condition2 = False
    while not condition2:
        rotate_joystick(joystick_pos)
        pyautogui.moveTo(*joystick_pos)
        condition2 = bool(find_on_screen(end_template, confidence=0.6))

    # 点击结算页面
    pyautogui.mouseUp(button='left')
    click_on_screen(*click_pos)
    time.sleep(0.5)