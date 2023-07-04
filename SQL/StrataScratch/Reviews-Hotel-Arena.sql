SELECT hotel_name, reviewer_score, COUNT(*) AS num_reviews
FROM hotel_reviews
WHERE hotel_name = 'Hotel Arena'
GROUP BY hotel_name, reviewer_score
ORDER BY num_reviews DESC
