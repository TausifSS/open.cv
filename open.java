import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;
import org.opencv.highgui.HighGui;

public class FaceDetection {
    static {
        // Load the OpenCV native library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {
        // Load the face detection classifier (Haar cascade)
        CascadeClassifier faceDetector = new CascadeClassifier("resources/haarcascade_frontalface_default.xmC:\\Users\\Tausif Shaikh\\IdeaProjects\\AI\\out\\resources\\haarcascade_frontalface_default.xmll");

        // Capture video from the webcam
        VideoCapture capture = new VideoCapture(0);

        if (!capture.isOpened()) {
            System.out.println("Error: Could not open video stream from webcam.");
            return;
        }

        Mat frame = new Mat();
        while (capture.read(frame)) {
            MatOfRect faceDetections = new MatOfRect();
            faceDetector.detectMultiScale(frame, faceDetections, 1.1, 10, 0, new Size(30, 30), new Size());

            // Draw rectangles around detected faces
            for (Rect rect : faceDetections.toArray()) {
                Imgproc.rectangle(frame, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(0, 255, 0));
            }

            // Display the result
            HighGui.imshow("Face Detection", frame);

            if (HighGui.waitKey(30) >= 0) {
                break;
            }
        }

        // Release the capture
        capture.release();
        HighGui.destroyAllWindows();
    }
}
