from ultralytics import YOLO

DATA_YAML = "C:/Users/pipkintj/cssema416/src/Project/cards_data/data.yaml"
IMAGE = IMAGE = "C:/Users/pipkintj/cssema416/src/Project/cards_data/test/images/001771721_jpg.rf.87d118febad97279b29b2b146da54d1a.jpg"
TRAIN = False                         

VALUES = {"A": 11, "J": 10, "Q": 10, "K": 10}
VALUES.update({str(i): i for i in range(2, 11)})

def hand_total(ranks):
    total = sum(VALUES[r] for r in ranks)
    aces = ranks.count("A")
    while total > 21 and aces:   
        total -= 10
        aces -= 1
    return total


if TRAIN:
    model = YOLO("yolo11n.pt")                      
    model.train(data=DATA_YAML, epochs=1, imgsz=640, fliplr=0.0)
    weights = "runs/detect/train-2/weights/best.pt"
else:
    weights = "runs/detect/train-2/weights/best.pt"

model = YOLO(weights)
result = model.predict(IMAGE, conf=0.4, agnostic_nms=True)[0]
height = result.orig_shape[0]

cards = []
for box in result.boxes:
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    name = result.names[int(box.cls)]    
    cards.append({"rank": name[:-1].upper(), "y": (y1 + y2) / 2})

cards.sort(key=lambda c: c["y"])         

dealer = [c["rank"] for c in cards[:1]]  
player = [c["rank"] for c in cards[1:]]  

print("Dealer:", dealer, "-> total", hand_total(dealer) if dealer else 0)
print("Player:", player, "-> total", hand_total(player) if player else 0)

result.save(filename="out.jpg")          
