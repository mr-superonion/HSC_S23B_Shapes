SELECT
  meas.object_id

, meas.g_cmodel_mag
, meas.g_cmodel_magerr
, meas.g_cmodel_flag

, meas.r_cmodel_mag
, meas.r_cmodel_magerr
, meas.r_cmodel_flag

, meas.z_cmodel_mag
, meas.z_cmodel_magerr
, meas.z_cmodel_flag

, meas.y_cmodel_mag
, meas.y_cmodel_magerr
, meas.y_cmodel_flag

, meas2.i_sdssshape_shape11
, meas2.i_sdssshape_shape22
, meas2.i_sdssshape_shape12

FROM
s23b_wide.meas as meas
LEFT JOIN s23b_wide.meas2 as meas2 using (object_id)

WHERE
meas.tract = {$tract}                       AND
NOT meas.g_pixelflags_edge                  AND
NOT meas.g_pixelflags_interpolatedcenter    AND
NOT meas.g_pixelflags_saturatedcenter       AND
NOT meas.g_pixelflags_crcenter              AND
NOT meas.g_pixelflags_bad                   AND
NOT meas.g_pixelflags_suspectcenter         AND
NOT meas.g_pixelflags_clipped               AND
NOT meas.r_pixelflags_edge                  AND
NOT meas.r_pixelflags_interpolatedcenter    AND
NOT meas.r_pixelflags_saturatedcenter       AND
NOT meas.r_pixelflags_crcenter              AND
NOT meas.r_pixelflags_bad                   AND
NOT meas.r_pixelflags_suspectcenter         AND
NOT meas.r_pixelflags_clipped               AND
NOT meas.i_pixelflags_edge                  AND
NOT meas.i_pixelflags_interpolatedcenter    AND
NOT meas.i_pixelflags_saturatedcenter       AND
NOT meas.i_pixelflags_crcenter              AND
NOT meas.i_pixelflags_bad                   AND
NOT meas.i_pixelflags_suspectcenter         AND
NOT meas.i_pixelflags_clipped               AND
NOT meas.z_pixelflags_edge                  AND
NOT meas.z_pixelflags_interpolatedcenter    AND
NOT meas.z_pixelflags_saturatedcenter       AND
NOT meas.z_pixelflags_crcenter              AND
NOT meas.z_pixelflags_bad                   AND
NOT meas.z_pixelflags_suspectcenter         AND
NOT meas.z_pixelflags_clipped               AND
NOT meas.y_pixelflags_edge                  AND
NOT meas.y_pixelflags_interpolatedcenter    AND
NOT meas.y_pixelflags_saturatedcenter       AND
NOT meas.y_pixelflags_crcenter              AND
NOT meas.y_pixelflags_bad                   AND
NOT meas.y_pixelflags_suspectcenter         AND
NOT meas.y_pixelflags_clipped               AND
meas.i_detect_isprimary                     AND
meas.i_extendedness_value != 0
ORDER BY meas.object_id
