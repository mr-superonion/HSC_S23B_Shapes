SELECT
  meas.object_id
, meas.parent_id
, meas.tract
, meas.patch
, meas.i_ra
, meas.i_dec
, meas.i_variance_value

-- unforced CModel magnitudes
, meas.i_cmodel_mag
, meas.i_cmodel_magerr
, meas.i_cmodel_flag
, meas.i_cmodel_flag_badcentroid
, meas.i_cmodel_flux
, meas.i_cmodel_fluxerr
, meas.i_cmodel_objective

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

-- forced measurement
, forced.a_g
, forced.a_r
, forced.a_i
, forced.a_z
, forced.a_y

-- forced Kron magnitudes
, forced2.g_kronflux_mag    as forced_g_kronflux_mag
, forced2.g_kronflux_magerr as forced_g_kronflux_magerr
, forced2.g_kronflux_flag   as forced_g_kronflux_flag

, forced2.r_kronflux_mag    as forced_r_kronflux_mag
, forced2.r_kronflux_magerr as forced_r_kronflux_magerr
, forced2.r_kronflux_flag   as forced_r_kronflux_flag

, forced2.i_kronflux_mag    as forced_i_kronflux_mag
, forced2.i_kronflux_magerr as forced_i_kronflux_magerr
, forced2.i_kronflux_flag   as forced_i_kronflux_flag

, forced2.z_kronflux_mag    as forced_z_kronflux_mag
, forced2.z_kronflux_magerr as forced_z_kronflux_magerr
, forced2.z_kronflux_flag   as forced_z_kronflux_flag

, forced2.y_kronflux_mag    as forced_y_kronflux_mag
, forced2.y_kronflux_magerr as forced_y_kronflux_magerr
, forced2.y_kronflux_flag   as forced_y_kronflux_flag

-- forced CModel magnitudes and fluxes
, forced.g_cmodel_mag       as forced_g_cmodel_mag
, forced.g_cmodel_magerr    as forced_g_cmodel_magerr
, forced.g_cmodel_flag      as forced_g_cmodel_flag

, forced.r_cmodel_mag       as forced_r_cmodel_mag
, forced.r_cmodel_magerr    as forced_r_cmodel_magerr
, forced.r_cmodel_flag      as forced_r_cmodel_flag

, forced.i_cmodel_mag       as forced_i_cmodel_mag
, forced.i_cmodel_magerr    as forced_i_cmodel_magerr
, forced.i_cmodel_flag      as forced_i_cmodel_flag

, forced.z_cmodel_mag       as forced_z_cmodel_mag
, forced.z_cmodel_magerr    as forced_z_cmodel_magerr
, forced.z_cmodel_flag      as forced_z_cmodel_flag

, forced.y_cmodel_mag       as forced_y_cmodel_mag
, forced.y_cmodel_magerr    as forced_y_cmodel_magerr
, forced.y_cmodel_flag      as forced_y_cmodel_flag

, meas2.i_psfflux_mag
, meas2.i_psfflux_magerr
, meas2.i_sdssshape_psf_shape11
, meas2.i_sdssshape_psf_shape22
, meas2.i_sdssshape_psf_shape12

-- Input exposures number
, meas.g_inputcount_value
, meas.r_inputcount_value
, meas.i_inputcount_value
, meas.z_inputcount_value
, meas.y_inputcount_value

-- Mask plane
, meas.i_pixelflags_inexact_psfcenter
, meas.i_pixelflags_inexact_psf

, meas4.g_convolvedflux_2_15_mag
, meas4.r_convolvedflux_2_15_mag
, meas4.i_convolvedflux_2_15_mag
, meas4.z_convolvedflux_2_15_mag
, meas4.y_convolvedflux_2_15_mag

, g_mask_brightstar_halo
, r_mask_brightstar_halo
, i_mask_brightstar_halo
, z_mask_brightstar_halo
, y_mask_brightstar_halo

, g_mask_brightstar_blooming
, r_mask_brightstar_blooming
, i_mask_brightstar_blooming
, z_mask_brightstar_blooming
, y_mask_brightstar_blooming

, g_mask_brightstar_ghost12
, r_mask_brightstar_ghost12
, i_mask_brightstar_ghost12
, z_mask_brightstar_ghost12
, y_mask_brightstar_ghost12

FROM
s23b_wide.meas as meas
LEFT JOIN s23b_wide.meas2 as meas2 using (object_id)
LEFT JOIN s23b_wide.meas3 as meas3 using (object_id)
LEFT JOIN s23b_wide.meas4 as meas4 using (object_id)
LEFT JOIN s23b_wide.forced as forced using (object_id)
LEFT JOIN s23b_wide.forced2 as forced2 using (object_id)
LEFT JOIN s23b_wide.masks as masks using (object_id)

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
