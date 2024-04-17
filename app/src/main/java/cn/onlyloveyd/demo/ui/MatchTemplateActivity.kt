package cn.onlyloveyd.demo.ui

import android.os.Bundle
import android.util.Log
import android.view.Menu
import android.view.MenuItem
import android.widget.RadioGroup
import androidx.appcompat.app.AppCompatActivity
import androidx.databinding.DataBindingUtil
import cn.onlyloveyd.demo.App
import cn.onlyloveyd.demo.R
import cn.onlyloveyd.demo.databinding.ActivityMatchTemplateBinding
import cn.onlyloveyd.demo.ext.ImageParser
import cn.onlyloveyd.demo.ext.showMat
import com.blankj.utilcode.util.FileUtils
import com.blankj.utilcode.util.ImageUtils
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc

/**
 * 模板匹配
 * author: yidong
 * 2020/10/23
 */
class MatchTemplateActivity : AppCompatActivity() {

    private lateinit var mBinding: ActivityMatchTemplateBinding
    private lateinit var mRgb: Mat
    private lateinit var mTemplate: Mat
    private var method = Imgproc.TM_SQDIFF
        set(value) {
            field = value
            doMatch(field)
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        mBinding = DataBindingUtil.setContentView(this, R.layout.activity_match_template)
        click()

        title = "TM_SQDIFF"
        // 目标图片
        val bgr = Utils.loadResource(this, R.drawable.test01)
        mRgb = Mat()
        mTemplate = Mat()
        Imgproc.cvtColor(bgr, mRgb, Imgproc.COLOR_BGR2RGB)
        // 模板
        val templateBgr = Utils.loadResource(this, R.drawable.test_tmp)
        Imgproc.cvtColor(templateBgr, mTemplate, Imgproc.COLOR_BGR2RGB)
        mBinding.ivLena.showMat(mTemplate)
        doMatch(method)

        // 使用新的方案测试
        match()

    }

    private fun match() {
        // val templateBitmap = ImageUtils.getBitmap(R.drawable.kobe_template)
        // val srcBitmap = ImageUtils.getBitmap(R.drawable.kobe)

        val templateBitmap = ImageUtils.getBitmap(R.drawable.test_tmp)
        val srcBitmap = ImageUtils.getBitmap(R.drawable.test01)
        ImageParser.drawMatches(srcBitmap,templateBitmap,null)
    }

    private fun click() {
        mBinding.rg.setOnCheckedChangeListener(RadioGroup.OnCheckedChangeListener { group, checkedId ->
            when (checkedId) {
                R.id.rb_tm_sqdiff -> {
                    method = Imgproc.TM_SQDIFF
                    title = "TM_SQDIFF"
                }
                R.id.rb_tm_sqdiff_normed -> {
                    method = Imgproc.TM_SQDIFF_NORMED
                    title = "TM_SQDIFF_NORMED"
                }
                R.id.rb_tm_ccoeff -> {
                    method = Imgproc.TM_CCOEFF
                    title = "TM_CCOEFF"
                }

                R.id.rb_tm_ccoeff_normed -> {
                    method = Imgproc.TM_CCOEFF_NORMED
                    title = "TM_CCOEFF_NORMED"
                }
                R.id.rb_tm_ccorr -> {
                    method = Imgproc.TM_CCORR
                    title = "TM_CCORR"
                }
                R.id.rb_tm_ccorr_normed -> {
                    method = Imgproc.TM_CCORR_NORMED
                    title = "TM_CCORR_NORMED"
                }
            }
        })
    }

    private fun doMatch(method: Int) {
        val tmp = mRgb.clone()
        val result = Mat()
        Imgproc.matchTemplate(mRgb, mTemplate, result, method)
        val minMaxLoc = Core.minMaxLoc(result)
        var info =
            "maxVal = ${minMaxLoc.maxVal}, maxLocation = ${minMaxLoc.maxLoc}, " +
                    "minVal = ${minMaxLoc.minVal}, minLocation = ${minMaxLoc.minLoc}"
        val topLeft = if (method == Imgproc.TM_SQDIFF || method == Imgproc.TM_SQDIFF_NORMED) {
            minMaxLoc.minLoc
        } else {
            minMaxLoc.maxLoc
        }
        info += "\n top-left: $topLeft"
        Log.d(App.TAG, info)
        mBinding.tvInfo.setText(info)
        val rect = Rect(topLeft, Size(mTemplate.cols().toDouble(), mTemplate.rows().toDouble()))
        Imgproc.rectangle(tmp, rect, Scalar(255.0, 0.0, 0.0), 4, Imgproc.LINE_8)
        mBinding.ivResult.showMat(tmp)
        tmp.release()
    }

    override fun onCreateOptionsMenu(menu: Menu?): Boolean {
        menuInflater.inflate(R.menu.menu_match_template, menu)
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        when (item.itemId) {
            R.id.match_tm_sqdiff -> {
                method = Imgproc.TM_SQDIFF
                title = "TM_SQDIFF"
            }
            R.id.match_tm_sqdiff_normed -> {
                method = Imgproc.TM_SQDIFF_NORMED
                title = "TM_SQDIFF_NORMED"
            }
            R.id.match_tm_ccoeff -> {
                method = Imgproc.TM_CCOEFF
                title = "TM_CCOEFF"
            }

            R.id.match_tm_ccoeff_normed -> {
                method = Imgproc.TM_CCOEFF_NORMED
                title = "TM_CCOEFF_NORMED"
            }
            R.id.match_tm_ccorr -> {
                method = Imgproc.TM_CCORR
                title = "TM_CCORR"
            }
            R.id.match_tm_ccorr_normed -> {
                method = Imgproc.TM_CCORR_NORMED
                title = "TM_CCORR_NORMED"
            }
        }
        return true
    }

    override fun onDestroy() {
        mTemplate.release()
        mRgb.release()
        super.onDestroy()
    }
}