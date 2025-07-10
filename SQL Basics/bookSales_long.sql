USE booksales;

SELECT * FROM newsletter;
SELECT * FROM web;
SELECT * FROM store;

-- compute purchases from the web for each user
SELECT UserID, ROUND(SUM(PurchaseAmount),2) AS Purchases_Online, COUNT(PurchaseAmount) AS Visits_Online
FROM web
GROUP BY UserID;

-- compute purchases from the store for each user
SELECT UserID, ROUND(SUM(PurchaseAmount),2) AS Purchases_Store, COUNT(PurchaseAmount) AS Visits_Store
FROM store
GROUP BY UserID;


-- combine online and store purchases
SELECT UserID, "Online" AS Location, ROUND(SUM(PurchaseAmount),2) AS Purchases, COUNT(PurchaseAmount) AS Visits
FROM web
GROUP BY UserID
UNION
SELECT UserID, "Store" AS Location, ROUND(SUM(PurchaseAmount),2) AS Purchases, COUNT(PurchaseAmount) AS Visits
FROM store
GROUP BY UserID
ORDER BY UserID;

-- combine with newsletter
SELECT summary.UserID, Location, Purchases, Visits,
		CASE
			WHEN Newsletter.UserID IS NULL THEN 0
            ELSE 1
		END AS Newsletter
FROM
(
	SELECT UserID, "Online" AS Location, ROUND(SUM(PurchaseAmount),2) AS Purchases, COUNT(PurchaseAmount) AS Visits
	FROM web
	GROUP BY UserID
	UNION
	SELECT UserID, "Store" AS Location, ROUND(SUM(PurchaseAmount),2) AS Purchases, COUNT(PurchaseAmount) AS Visits
	FROM store
	GROUP BY UserID
) AS summary
LEFT JOIN Newsletter ON summary.UserID = Newsletter.UserID
ORDER BY UserID;

CREATE TABLE salesdw_long AS
SELECT summary.UserID, Location, Purchases, Visits,
		CASE
			WHEN Newsletter.UserID IS NULL THEN 0
            ELSE 1
		END AS Newsletter
FROM
(
	SELECT UserID, "Online" AS Location, ROUND(SUM(PurchaseAmount),2) AS Purchases, COUNT(PurchaseAmount) AS Visits
	FROM web
	GROUP BY UserID
	UNION
	SELECT UserID, "Store" AS Location, ROUND(SUM(PurchaseAmount),2) AS Purchases, COUNT(PurchaseAmount) AS Visits
	FROM store
	GROUP BY UserID
) AS summary
LEFT JOIN Newsletter ON summary.UserID = Newsletter.UserID
ORDER BY UserID;

SELECT * FROM  salesdw_long;

-- statistics by running queries on data warehouse 
-- compare online and in store
SELECT 	Newsletter, Location,
		SUM(Purchases) AS Purchases_Total, SUM(Visits) AS Visits, SUM(Purchases)/SUM(Visits) AS Visit_Avg
FROM salesdw_long
GROUP BY Newsletter, Location WITH ROLLUP;

-- frequency of visits online 
SELECT Visits, COUNT(*) AS Freq
FROM salesdw_long
WHERE Location = "Online"
GROUP BY Visits WITH ROLLUP;

-- frequency of visits online and in store 
SELECT Location, Visits, COUNT(*) AS Freq
FROM salesdw_long
GROUP BY Location, Visits WITH ROLLUP;






