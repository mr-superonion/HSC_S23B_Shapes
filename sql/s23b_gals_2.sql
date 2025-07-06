SELECT
  meas.object_id
, meas.tract
, meas.patch
, meas.i_ra
, meas.i_dec
, meas.i_variance_value

, meas.g_cmodel_mag
, meas.g_cmodel_magerr
, meas.g_cmodel_flag
, meas.r_cmodel_mag
, meas.r_cmodel_magerr
, meas.r_cmodel_flag
, meas.i_cmodel_mag
, meas.i_cmodel_magerr
, meas.i_cmodel_flag
, meas.z_cmodel_mag
, meas.z_cmodel_magerr
, meas.z_cmodel_flag
, meas.y_cmodel_mag
, meas.y_cmodel_magerr
, meas.y_cmodel_flag

, meas2.g_psfflux_mag
, meas2.g_psfflux_magerr
, meas2.r_psfflux_mag
, meas2.r_psfflux_magerr
, meas2.i_psfflux_mag
, meas2.i_psfflux_magerr
, meas2.z_psfflux_mag
, meas2.z_psfflux_magerr
, meas2.y_psfflux_mag
, meas2.y_psfflux_magerr

, f1.a_g
, f1.a_r
, f1.a_i
, f1.a_z
, f1.a_y

, meas2.i_blendedness_abs
, meas5.i_hsmpsfmoments_shape11
, meas5.i_hsmpsfmoments_shape22
, meas5.i_hsmpsfmoments_shape12
, meas5.i_higherordermomentspsf_04
, meas5.i_higherordermomentspsf_13
, meas5.i_higherordermomentspsf_22
, meas5.i_higherordermomentspsf_31
, meas5.i_higherordermomentspsf_40

FROM
s23b_wide.meas as meas
LEFT JOIN s23b_wide.forced as f1 using (object_id)
LEFT JOIN s23b_wide.meas2 as meas2 using (object_id)
LEFT JOIN s23b_wide.meas5 as meas5 using (object_id)

WHERE
meas.tract = {$tract}                       AND
meas.i_detect_isprimary
ORDER BY meas.object_id
