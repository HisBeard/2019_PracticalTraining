from face_tool import detect_image

if __name__ == '__main__':
    result = detect_image("test1.jpg", "output1.jpg")
    print(isinstance(result, dict))
    detect_image("test2.jpg", "output2.jpg")
    detect_image("test3.jpg", "output3.jpg")
