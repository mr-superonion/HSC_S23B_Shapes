SELECT
    object_id
    ,m1.tract
    ,m1.patch
    ,m1.g_inputcount_value
    ,m1.r_inputcount_value
    ,m1.i_inputcount_value
    ,m1.z_inputcount_value
    ,m1.y_inputcount_value
    ,m1.g_variance_value
    ,m1.r_variance_value
    ,m1.i_variance_value
    ,m1.z_variance_value
    ,m1.y_variance_value
    ,m1.i_pixelflags_inexact_psfcenter
    ,m1.i_pixelflags_inexact_psf
    ,m5.g_hsmpsfmoments_shape11
    ,m5.g_hsmpsfmoments_shape22
    ,m5.g_hsmpsfmoments_shape12
    ,m5.r_hsmpsfmoments_shape11
    ,m5.r_hsmpsfmoments_shape22
    ,m5.r_hsmpsfmoments_shape12
    ,m5.i_hsmpsfmoments_shape11
    ,m5.i_hsmpsfmoments_shape22
    ,m5.i_hsmpsfmoments_shape12
    ,m5.z_hsmpsfmoments_shape11
    ,m5.z_hsmpsfmoments_shape22
    ,m5.z_hsmpsfmoments_shape12
    ,m5.y_hsmpsfmoments_shape11
    ,m5.y_hsmpsfmoments_shape22
    ,m5.y_hsmpsfmoments_shape12
FROM
    s23b_wide.forced  AS f1
  LEFT JOIN
    s23b_wide.meas  AS m1 USING (object_id)
  LEFT JOIN
    s23b_wide.meas2 AS m2 using (object_id)
  LEFT JOIN
    s23b_wide.meas5 AS m5 using (object_id)
WHERE
    m1.tract = {$tract}                     AND
    f1.isprimary                            AND
    m1.i_calib_psf_candidate                AND
    NOT m1.i_pixelflags_edge                AND
    NOT m1.i_pixelflags_interpolatedcenter  AND
    NOT m1.i_pixelflags_saturatedcenter     AND
    NOT m1.i_pixelflags_crcenter            AND
    NOT m1.i_pixelflags_bad                 AND
    NOT m1.i_pixelflags_suspectcenter       AND
    NOT m1.i_pixelflags_clipped
;
