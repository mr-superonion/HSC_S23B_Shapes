SELECT

ran.ra
, ran.dec
, ran.tract
, ran.patch
, ran.g_inputcount_value
, ran.r_inputcount_value
, ran.i_inputcount_value
, ran.z_inputcount_value
, ran.y_inputcount_value

FROM
s23b_wide.random as ran

WHERE
ran.tract = {$tract}                       AND
ran.isprimary

ORDER BY ran.object_id
