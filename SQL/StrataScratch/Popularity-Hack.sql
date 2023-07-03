SELECT fe.location, AVG(fhs.popularity) AS avg_popularity
FROM facebook_employees AS fe
INNER JOIN facebook_hack_survey AS fhs
on fe.id = fhs.employee_id
GROUP BY location
