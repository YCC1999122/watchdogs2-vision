"""ImageNet-1000 fine classifier: relabels YOLO detections with a much wider
vocabulary (teacup, dish antenna, ...) and fixes objects mistaken as person."""
import os

import cv2
import numpy as np

# common ImageNet classes -> Chinese (fallback: English name shown as-is)
ZH = {
    "cup": "茶杯", "coffee mug": "咖啡杯", "teapot": "茶壶", "beer glass": "啤酒杯",
    "wine bottle": "酒瓶", "beer bottle": "啤酒瓶", "water bottle": "水瓶",
    "pop bottle": "汽水瓶", "vase": "花瓶", "bowl": "碗", "soup bowl": "汤碗",
    "plate": "盘子", "frying pan": "平底锅", "wok": "炒锅", "pot": "锅",
    "dish aerial": "卫星天线", "radio": "收音机", "television": "电视机",
    "monitor": "显示器", "screen": "屏幕", "laptop": "笔记本电脑",
    "desktop computer": "台式电脑", "notebook": "笔记本电脑", "iPod": "音乐播放器",
    "cellular telephone": "手机", "telephone": "电话", "dial telephone": "座机",
    "remote control": "遥控器", "mouse": "鼠标", "keyboard": "键盘",
    "space bar": "键盘", "printer": "打印机", "photocopier": "复印机",
    "hard disc": "硬盘", "modem": "调制解调器", "loudspeaker": "音箱",
    "microphone": "麦克风", "headphone": "耳机", "camera": "相机",
    "reflex camera": "单反相机", "lens cap": "镜头盖", "projector": "投影仪",
    "microwave": "微波炉", "oven": "烤箱", "toaster": "烤面包机",
    "refrigerator": "冰箱", "washer": "洗衣机", "Dutch oven": "炖锅",
    "espresso maker": "咖啡机", "vacuum": "吸尘器", "fan": "风扇",
    "electric fan": "电风扇", "table lamp": "台灯", "floor lamp": "落地灯",
    "umbrella": "雨伞", "backpack": "背包", "handbag": "手提包", "purse": "钱包",
    "suitcase": "行李箱", "briefcase": "公文包", "tie": "领带", "sunglasses": "墨镜",
    "sunglass": "太阳镜", "hat": "帽子", "cowboy hat": "牛仔帽", "helmet": "头盔",
    "crash helmet": "头盔", "bicycle": "自行车", "motorcycle": "摩托车",
    "car": "汽车", "sports car": "跑车", "racer": "赛车", "truck": "卡车",
    "fire engine": "消防车", "ambulance": "救护车", "bus": "公交车",
    "minibus": "小巴", "taxi": "出租车", "trailer truck": "挂车",
    "airliner": "客机", "airship": "飞艇", "boat": "船", "canoe": "独木舟",
    "pirate": "帆船", "speedboat": "快艇", "catamaran": "双体船",
    "traffic light": "红绿灯", "street sign": "路牌", "parking meter": "停车计时器",
    "fire hydrant": "消防栓", "mailbox": "邮箱", "bench": "长椅", "chair": "椅子",
    "rocking chair": "摇椅", "throne": "王座", "table": "桌子", "desk": "书桌",
    "dining table": "餐桌", "bookcase": "书柜", "bookshelf": "书架",
    "wardrobe": "衣柜", "bed": "床", "couch": "沙发", "studio couch": "沙发床",
    "toilet seat": "马桶", "bathtub": "浴缸", "shower": "淋浴",
    "book": "书", "comic book": "漫画书", "notebook (paper)": "笔记本",
    "envelope": "信封", "carton": "纸箱", "crate": "板条箱", "barrel": "桶",
    "basket": "篮子", "bucket": "水桶", "ashcan": "垃圾桶",
    "teddy": "泰迪熊", "toy": "玩具", "ball": "球", "soccer ball": "足球",
    "basketball": "篮球", "tennis ball": "网球", "baseball": "棒球",
    "golf ball": "高尔夫球", "kite": "风筝", "balloon": "气球",
    "dog": "狗", "cat": "猫", "bird": "鸟", "horse": "马", "sheep": "羊",
    "cow": "牛", "elephant": "大象", "zebra": "斑马", "giraffe": "长颈鹿",
    "monkey": "猴子", "rabbit": "兔子", "squirrel": "松鼠", "fox": "狐狸",
    "bottle opener": "开瓶器", "screwdriver": "螺丝刀", "hammer": "锤子",
    "wrench": "扳手", "pliers": "钳子", "drill": "电钻", "ladder": "梯子",
    "scissors": "剪刀", "knife": "刀", "fork": "叉", "spoon": "勺子",
    "banana": "香蕉", "apple": "苹果", "orange": "橙子", "pizza": "披萨",
    "hotdog": "热狗", "sandwich": "三明治", "cake": "蛋糕", "doughnut": "甜甜圈",
    "ice cream": "冰淇淋", "ice lolly": "冰棍", "coffee": "咖啡",
    "person": "人", "man": "男人", "woman": "女人", "child": "儿童",
    "groom": "男人", "scuba diver": "潜水员",
}

# ImageNet classes that count as "electronic device" for blue/red status logic
ELECTRONIC = {
    "radio", "television", "monitor", "screen", "laptop", "desktop computer",
    "notebook", "iPod", "cellular telephone", "telephone", "dial telephone",
    "remote control", "mouse", "keyboard", "printer", "photocopier",
    "hard disc", "modem", "loudspeaker", "microphone", "headphone", "camera",
    "reflex camera", "projector", "microwave", "oven", "toaster",
    "refrigerator", "washer", "vacuum", "fan", "electric fan", "CD player",
    "tape player", "space heater", "oscilloscope", "pay-phone", "joystick",
}
PERSONISH = {"person", "man", "woman", "child", "groom", "scuba diver"}

# clothing / things worn by people: a person crop often classifies as these —
# they must NOT cause a person detection to be demoted to an object
CLOTHING = {
    "jersey", "T-shirt", "sweatshirt", "cardigan", "suit", "trench coat",
    "pajama", "miniskirt", "jean", "denim", "sock", "running shoe", "sandal",
    "clog", "loafer", "cowboy boot", "mitten", "glove", "wig", "sombrero",
    "bonnet", "bow tie", "Windsor tie", "wool", "maillot", "bikini",
    "brassiere", "diaper", "gown", "robe", "academic gown", "apron", "bib",
    "kimono", "abaya", "sarong", "poncho", "parka", "anorak", "vestment",
    "lab coat", "fur coat", "stole", "ski mask", "swimming trunks",
    "military uniform", "vestment", "bow tie", "shower cap", "bathing cap",
    "crash helmet", "football helmet", "pickelhaube", "cowboy hat",
    "sunglasses", "sunglass", "mask", "gas mask", "oxygen mask",
    "backpack", "handbag", "purse", "neck brace", "umbrella",
    "holster", "chain mail", "cuirass", "bulletproof vest", "velvet",
    "fur", "sneaker", "slipper", "boot", "shoe", "cap", "hat", "helmet",
}


class FineLabeler:
    def __init__(self, model_path, labels_path):
        self.ok = os.path.exists(model_path) and os.path.exists(labels_path)
        if not self.ok:
            return
        self.net = cv2.dnn.readNetFromONNX(model_path)
        with open(labels_path, "r", encoding="utf-8") as f:
            self.names = [l.strip() for l in f if l.strip()]

    def classify(self, bgr):
        """-> (name_en, conf). MobileNetV2 224x224."""
        blob = cv2.dnn.blobFromImage(
            bgr, 1 / 255.0, (224, 224),
            (0.485 * 255, 0.456 * 255, 0.406 * 255), swapRB=True)
        blob = blob / np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)
        self.net.setInput(blob)
        out = self.net.forward().flatten()
        e = np.exp(out - out.max())
        prob = e / e.sum()
        i = int(prob.argmax())
        name = self.names[i] if i < len(self.names) else f"cls{i}"
        return name, float(prob[i])


def display_name(en_name):
    """English ImageNet name -> Chinese if we know it."""
    return ZH.get(en_name, en_name)
