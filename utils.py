import cv2
import numpy as np
from tflite_support.task import processor
from tracker import *
import threading
import time
import serial
import queue

_MARGIN = 10  # pixels
_ROW_SIZE = 10  # pixels
_FONT_SIZE = 1
_FONT_THICKNESS = 1
_TEXT_COLOR = (255, 0, 0)  # red
tracker=Tracker()
area=[(145,2),(145,474),(160,474),(160,2)]
area1_c=set()
queue= queue.Queue()
# Configure the serial port
lora = serial.Serial(
    port='/dev/serial0',  # Raspberry Pi UART port
    baudrate=9600,         # Match baudrate with your LoRa module
    timeout=1              # Timeout for serial communication
)
#to get lora address
lora.write(b'AT+ADDRESS?\r\n')
address = lora.readline().decode().strip()
address= address[9:]
address= int(address)
print(address)
#Function to listen for LoRa msg
def lora_message():
    remsg = lora.readline().decode().strip()
    #remsg ='+RCV=2,1,2,-50,44'
    if remsg.startswith('+RCV'):
        parts = remsg.split(',')
        if len(parts) >= 3:  # Check if there are enough elements after splitting
            per_cars= queue.put(int(parts[2]))
            print(per_cars)
# Function to send the number of cars  
def send_num_cars():
    while True:
        #set time interval
        time.sleep(60)
        num_cars = len(area1_c)
        # Send data to LoRa module
        msg = f'AT+SEND={address+1},2,{num_cars}\r\n'
        msg = msg.encode('utf-8')
        lora.write(msg)
        # Wait for a response
        response = lora.readline().decode().strip()
        print(response)
        print("Number of cars:", num_cars)
        pre_cars=queue.get()
        if num_cars < pre_cars:
            #send emergency message to the drone
            lora.write(b'AT+SEND=1,3,SOS\r\n')
            response = lora.readline().decode().strip()
            print(response)
            print('sos')
# Start the thread for printing number of cars
send_thread = threading.Thread(target=send_num_cars)
send_thread.start()
lora_thread = threading.Thread(target=lora_message)
lora_thread.start()
def visualize(
    image: np.ndarray,
    detection_result: processor.DetectionResult,
) -> np.ndarray:
  """Draws bounding boxes on the input image and return it.

  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.

  Returns:
    Image with bounding boxes.
  """
  list=[]
  for detection in detection_result.detections:

    
    #detect cars only
    category = detection.categories[0]
    category_name = category.category_name
    if category_name =='car' or category_name=='truck':
        
        # Draw bounding_box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        x,y=start_point
        x1,y1=end_point
        list.append([x,y,x1,y1])
    
        # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (_MARGIN + bbox.origin_x,
                         _MARGIN + _ROW_SIZE + bbox.origin_y)
        bbox_idx=tracker.update(list)
        for bbox in bbox_idx:
            x2,y2,x3,y3,id=bbox
            cx=int(x2+x3)//2
            cy=int(y2+y3)//2
            results1=cv2.pointPolygonTest(np.array(area,np.int32),((cx,cy)),False)
            if results1>=0:
                 cv2.circle(image,(cx,cy),4,(255,0,255),-1)
                 cv2.rectangle(image,(x2,y2),(x3,y3),(0,255,0),2)
                 cv2.putText(image, str(id), (x2,y2), cv2.FONT_HERSHEY_PLAIN,
                        _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)
                 area1_c.add(id)
  num_cars= len(area1_c)


      
  cv2.putText(image, str(num_cars), (30,40), cv2.FONT_HERSHEY_PLAIN,2, (255,255,255), 2)
  cv2.polylines(image,[np.array(area,np.int32)],True,(255,0,0),2)

  return image

