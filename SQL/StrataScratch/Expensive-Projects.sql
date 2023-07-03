SELECT mp.title AS project, ROUND((budget/COUNT(mep.emp_id)::FLOAT)::NUMERIC, 0) AS buget_ratio
FROM ms_projects AS mp 
INNER JOIN ms_emp_projects AS mep
ON mp.id = mep.project_id
GROUP BY title, budget
ORDER BY buget_ratio DESC;
