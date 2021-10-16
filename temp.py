import cv2

cap = cv2.VideoCapture(0)

while True:
    frame, ret = cap.read()

    if cv2.waitKey(1) == ord('q'):
        break

    cv2.imshow("image",frame)

cv2.destroyAllWindows()
cap.release()