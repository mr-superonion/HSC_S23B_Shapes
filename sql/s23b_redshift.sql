SELECT
  meas.object_id
, meas.tract
, meas.patch
, mizuki.photoz_best
, mizuki.photoz_mc
, mizuki.photoz_err95_min
, mizuki.photoz_err95_max
, mizuki.photoz_err68_min
, mizuki.photoz_err68_max

FROM
s23b_wide.meas as meas
LEFT JOIN s23b_wide.photoz_mizuki as mizuki using (object_id)

WHERE
meas.tract = {$tract}                       AND
meas.i_detect_isprimary
ORDER BY meas.object_id
