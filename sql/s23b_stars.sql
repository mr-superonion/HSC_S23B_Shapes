SELECT
        object_id
        ,f1.ra
        ,f1.dec
        ,f2.g_psfflux_flux
        ,f2.g_psfflux_fluxerr
        ,f2.r_psfflux_flux
        ,f2.r_psfflux_fluxerr
        ,f2.i_psfflux_flux
        ,f2.i_psfflux_fluxerr
        ,f2.z_psfflux_flux
        ,f2.z_psfflux_fluxerr
        ,f2.y_psfflux_flux
        ,f2.y_psfflux_fluxerr
		,f2.i_sdssshape_shape11
		,f2.i_sdssshape_shape22
		,f2.i_sdssshape_shape12
		,f2.i_sdssshape_psf_shape11
		,f2.i_sdssshape_psf_shape22
		,f2.i_sdssshape_psf_shape12
		,m1.i_calib_psf_candidate
		,m1.i_calib_psf_reserved
		,m1.i_calib_psf_used
		,m1.tract
		,m1.patch
        ,f1.a_g
        ,f1.a_r
        ,f1.a_i
        ,f1.a_z
        ,f1.a_y
        ,m2.i_sdsscentroid_ra
        ,m2.i_sdsscentroid_dec
    FROM
        s23_wide.forced  AS f1
      LEFT JOIN
        s23_wide.forced2 AS f2 USING (object_id)
      LEFT JOIN
        s23_wide.meas2   AS m2 USING (object_id)
	  LEFT JOIN
        s23_wide.meas    AS m1 USING (object_id)
	WHERE
        f1.isprimary
		AND m1.i_calib_psf_candidate
;
