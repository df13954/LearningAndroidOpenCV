package cn.onlyloveyd.demo.ext;

import android.graphics.Bitmap;

import com.blankj.utilcode.util.CollectionUtils;
import com.blankj.utilcode.util.ImageUtils;
import com.blankj.utilcode.util.LogUtils;

import org.opencv.android.Utils;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.DMatch;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.features2d.BRISK;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.FlannBasedMatcher;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import cn.onlyloveyd.demo.R;

/**
 * Create by os on 2024/4/11
 * Desc :
 */
public class ImageParser {
    private static final String TAG = "ImageParser";

    /**
     * @param image2Bitmap   截图出来的Bitmap对象
     * @param templateBitmap 要匹配的按钮的Bitmap对象
     * @param template       要匹配的按钮的Mat对象
     */
    public static void drawMatches(Bitmap image2Bitmap, Bitmap templateBitmap, Mat template) {
        // image2Bitmap：截图出来的Bitmap对象
        // templateBitmap：要匹配的按钮的Bitmap对象
        // src：截图的Mat对象
        // template：要匹配的按钮的Mat对象
        Mat src;
        if (image2Bitmap != null) {
            LogUtils.d(TAG, "截图完成,准备匹配截图");
            src = new Mat();
            Utils.bitmapToMat(image2Bitmap, src);
            LogUtils.d(TAG, "获取的截图宽高", image2Bitmap.getWidth(), image2Bitmap.getHeight());
            if (templateBitmap == null) {
                templateBitmap = ImageUtils.getBitmap(R.drawable.kobe_template);
            }

            if (template == null) {
                template = new Mat();
                Utils.bitmapToMat(templateBitmap, template);
            }

            MatOfKeyPoint keyPointTemplate = new MatOfKeyPoint();
            Mat templateDescriptorMat = new Mat();

            MatOfKeyPoint keyPointSrc = new MatOfKeyPoint();
            Mat srcDescriptorMat = new Mat();

            //            ORB me = ORB.create(1000, 1.2f);
            //            ORB me = ORB.create();
            //            KAZE me = KAZE.create();
            //            AKAZE me = AKAZE.create();
            BRISK me = BRISK.create();
            //            MSER me = MSER.create();

            me.detect(template, keyPointTemplate);
            me.compute(template, keyPointTemplate, templateDescriptorMat);

            me.detect(src, keyPointSrc);
            me.compute(src, keyPointSrc, srcDescriptorMat);

            if (templateDescriptorMat.type() != CvType.CV_32F && srcDescriptorMat.type() != CvType.CV_32F) {
                templateDescriptorMat.convertTo(templateDescriptorMat, CvType.CV_32F);
                srcDescriptorMat.convertTo(srcDescriptorMat, CvType.CV_32F);
            }
            MatOfDMatch matches = new MatOfDMatch();
            FlannBasedMatcher matcher = FlannBasedMatcher.create();
            matcher.match(templateDescriptorMat, srcDescriptorMat, matches);

            List<DMatch> matchList = matches.toList();
            LogUtils.d(TAG, "优化前的匹配点数量：", matchList.size());
            // 按照distance升序
            Collections.sort(matchList, (a, b) -> {
                return (int) (a.distance * 1000 - b.distance * 1000);
            });
            LogUtils.d(TAG, "排序后的匹配点列表", matchList);
            // float min = matchList.get(0).distance;
            if (matchList.isEmpty()) {
                LogUtils.e(TAG, "无法匹配");
                return;
            }
            float max = matchList.get(matchList.size() - 1).distance;
            List<DMatch> goodMatchList = new ArrayList(matchList);
            // 对列表进行筛选，去除distance小于（最大的distance * 0.4）的特征点
            CollectionUtils.filter(goodMatchList, new CollectionUtils.Predicate<DMatch>() {
                @Override
                public boolean evaluate(DMatch item) {
                    return item.distance < max * 0.5;
                }
            });
            LogUtils.w(TAG, "优化后的匹配点数量：", goodMatchList.size());
            LogUtils.w(TAG, "优化后的匹配点数量：", goodMatchList.toString());
            // 如果匹配点小于4个，是无法继续执行的话，不然OpenCV会报错
            if (goodMatchList.size() < 4) {
                LogUtils.e(TAG, "匹配点小于4个，跳过本次！" +
                        "(如果匹配点小于4个，是无法继续执行的话，不然OpenCV会报错)");
                return;
            }

            // 承载最终结果的Mat
            Mat result = new Mat();
            MatOfDMatch matOfDMatch = new MatOfDMatch();
            matOfDMatch.fromList(goodMatchList);
            // 把匹配图在大图中的特征点关系线画上
            Features2d.drawMatches(template, keyPointTemplate, src, keyPointSrc, matOfDMatch, result);
            // 以上其实已经能标识出大图中关于匹配图的特征点关系了（用线连接），下面要做的是找出匹配图在大图中的位置

            //-- 定位对象
            List<Point> obj = new ArrayList<>();
            List<Point> scene = new ArrayList<>();
            List<KeyPoint> listOfKeypointsObject = keyPointTemplate.toList();
            List<KeyPoint> listOfKeypointsScene = keyPointSrc.toList();
            for (int i = 0; i < goodMatchList.size(); i++) {
                //-- 从良好的匹配中获取关键点
                obj.add(listOfKeypointsObject.get(goodMatchList.get(i).queryIdx).pt);
                scene.add(listOfKeypointsScene.get(goodMatchList.get(i).trainIdx).pt);
            }
            MatOfPoint2f objMat = new MatOfPoint2f();
            MatOfPoint2f sceneMat = new MatOfPoint2f();
            objMat.fromList(obj);
            sceneMat.fromList(scene);
            LogUtils.d(TAG, "listOfKeypointsScene point", listOfKeypointsScene.toString());

            // 输出匹配特征在源图像上的坐标
            for (Point point : scene) {
                LogUtils.d(TAG, "匹配特征点坐标：X = " + point.x + ", Y = " + point.y);
            }


            double ransacReprojThreshold = 3.0;
            Mat H = Calib3d.findHomography(objMat, sceneMat, Calib3d.RANSAC, ransacReprojThreshold);

            // LogUtils.d(TAG, "obj point", obj.toString());
            // LogUtils.d(TAG, "scene point", obj.toString());
            //-- 从image_1（要“检测”的对象）获取角
            Mat objCorners = new Mat(4, 1, CvType.CV_32FC2), sceneCorners = new Mat();
            float[] objCornersData = new float[(int) (objCorners.total() * objCorners.channels())];
            objCorners.get(0, 0, objCornersData);
            objCornersData[0] = 0;
            objCornersData[1] = 0;
            objCornersData[2] = template.cols();
            objCornersData[3] = 0;
            objCornersData[4] = template.cols();
            objCornersData[5] = template.rows();
            objCornersData[6] = 0;
            objCornersData[7] = template.rows();
            objCorners.put(0, 0, objCornersData);
            Core.perspectiveTransform(objCorners, sceneCorners, H);
            float[] sceneCornersData = new float[(int) (sceneCorners.total() * sceneCorners.channels())];
            sceneCorners.get(0, 0, sceneCornersData);
            LogUtils.d(TAG, "sceneCornersData", sceneCornersData);
            // 画框
            // -- 在角之间绘制线，也就是我要找的按钮的四个边
            Imgproc.line(result, new Point(sceneCornersData[0] + template.cols(), sceneCornersData[1]),
                    new Point(sceneCornersData[2] + template.cols(), sceneCornersData[3]), new Scalar(255, 0, 0, 255), 4);
            Imgproc.line(result, new Point(sceneCornersData[2] + template.cols(), sceneCornersData[3]),
                    new Point(sceneCornersData[4] + template.cols(), sceneCornersData[5]), new Scalar(255, 0, 0, 255), 4);
            Imgproc.line(result, new Point(sceneCornersData[4] + template.cols(), sceneCornersData[5]),
                    new Point(sceneCornersData[6] + template.cols(), sceneCornersData[7]), new Scalar(255, 0, 0, 255), 4);
            Imgproc.line(result, new Point(sceneCornersData[6] + template.cols(), sceneCornersData[7]),
                    new Point(sceneCornersData[0] + template.cols(), sceneCornersData[1]), new Scalar(255, 0, 0, 255), 4);

            Bitmap bitmap = Bitmap.createBitmap(result.width(), result.height(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(result, bitmap);
            // 保存图片到本地
            ImageUtils.save2Album(bitmap, Bitmap.CompressFormat.PNG);
            LogUtils.i(TAG, "截图保存到相册");

            // Core.MinMaxLocResult minMaxLoc = Core.minMaxLoc(result);
            // if (minMaxLoc != null) {
            //     LogUtils.i(TAG, "minMaxLoc: ", "maxVal =", minMaxLoc.maxVal,
            //             "", minMaxLoc.maxLoc.toString(), "minVal =", minMaxLoc.minVal, "", minMaxLoc.minLoc.toString());
            // }

        } else {
            LogUtils.d(TAG, "截图的Bitmap是空！！！");
        }
    }
}
