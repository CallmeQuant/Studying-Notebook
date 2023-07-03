SELECT to_user AS user_id, label, 
COUNT(*) AS num_occurences
FROM google_gmail_emails as ge
INNER JOIN google_gmail_labels as gl
ON ge.id = gl.email_id 
AND gl.label ILIKE 'custom%'
GROUP BY to_user, label;
