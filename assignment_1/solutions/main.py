import cv2



image = cv2.imread('../iris-1.jpg', 1)
camera = cv2.VideoCapture(0)



def print_image_information(image):
    print("Height: ", image.shape[0])
    print("Width: ", image.shape[1])
    print("Channels: ", image.shape[2])
    print("Size: ", image.size)
    print("Data Type: ", image.dtype)

def print_camera_information(camera):
    fps = int(camera.get(cv2.CAP_PROP_FPS))
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))

    with open("camera_outputs.txt", "w") as file:
        file.write(f"fps: {fps}\n")
        file.write(f"height: {height}\n")
        file.write(f"width: {width}\n")



print_image_information(image)
print_camera_information(camera)



camera.release()