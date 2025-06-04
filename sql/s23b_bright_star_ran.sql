SELECT

ran.ra
, ran.dec
, ran.tract
, ran.patch

FROM
s23b_wide.random as ran
LEFT JOIN s23b_wide.random_masks as ran_masks USING (object_id)

WHERE
ran.tract = {$tract}                       AND
ran_masks.g_mask_brightstar_blooming        OR
ran_masks.g_mask_brightstar_ghost           OR
ran_masks.r_mask_brightstar_blooming        OR
ran_masks.r_mask_brightstar_ghost           OR
ran_masks.i_mask_brightstar_halo            OR
ran_masks.i_mask_brightstar_blooming        OR
ran_masks.i_mask_brightstar_ghost           OR
ran_masks.z_mask_brightstar_blooming        OR
ran_masks.z_mask_brightstar_ghost           OR
ran_masks.y_mask_brightstar_blooming        OR
ran_masks.y_mask_brightstar_ghost

ORDER BY ran.object_id
