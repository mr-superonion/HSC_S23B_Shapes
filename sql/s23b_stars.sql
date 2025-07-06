SELECT
    object_id
    ,m1.i_ra
    ,m1.i_dec
    ,m1.tract
    ,m1.patch
    ,m1.i_variance_value
    ,m2.i_psfflux_flux
    ,m2.i_psfflux_fluxerr
    ,m1.i_calib_psf_reserved
    ,m1.i_calib_psf_used
    ,f1.a_i
    ,m5.i_hsmpsfmoments_shape11
    ,m5.i_hsmpsfmoments_shape22
    ,m5.i_hsmpsfmoments_shape12
	,m5.i_hsmsourcemoments_shape11
    ,m5.i_hsmsourcemoments_shape22
    ,m5.i_hsmsourcemoments_shape12
    ,m5.i_higherordermomentspsf_04
    ,m5.i_higherordermomentspsf_13
    ,m5.i_higherordermomentspsf_22
    ,m5.i_higherordermomentspsf_31
    ,m5.i_higherordermomentspsf_40
    ,m5.i_higherordermomentssource_04
    ,m5.i_higherordermomentssource_13
    ,m5.i_higherordermomentssource_22
    ,m5.i_higherordermomentssource_31
    ,m5.i_higherordermomentssource_40
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
