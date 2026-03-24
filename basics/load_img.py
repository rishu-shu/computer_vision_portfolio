import cv2

#LOAD IMAGE
img=cv2.imread("test.jpg")

#DISPLAY IMAGE
cv2.imshow("IMAGE",img)

cv2.waitKey(0)
cv2.destroyAllWindows()

