USE booksales;

SELECT * FROM newsletter;
SELECT * FROM web;
SELECT * FROM store;

-- 
SELECT UserID, ROUND(SUM(PurchaseAmount),2) AS Purchases_Online, COUNT(PurchaseAmount) AS Visits_Online
FROM web
GROUP BY UserID;

SELECT UserID, ROUND(SUM(PurchaseAmount),2) AS Purchases_Store, COUNT(PurchaseAmount) AS Visits_Store
FROM store
GROUP BY UserID;


-- combine purchases online with purchases in store for users who visited only online at least once
-- coalesce function will replace Null values with zeros
SELECT 	web.UserID, 
		COALESCE(Purchases_Online, 0) AS Purchases_Online,
        COALESCE(Visits_Online, 0) AS Visits_Online,
        COALESCE(Purchases_Store, 0) AS Purchases_Store,
        COALESCE(Visits_Store, 0) AS Visits_Store
FROM
(
	SELECT UserID, ROUND(SUM(PurchaseAmount),2) AS Purchases_Online, COUNT(PurchaseAmount) AS Visits_Online
	FROM web
	GROUP BY UserID
) AS web
LEFT JOIN 
	(
		SELECT UserID, ROUND(SUM(PurchaseAmount),2) AS Purchases_Store, COUNT(PurchaseAmount) AS Visits_Store
		FROM store
		GROUP BY UserID
	) AS store
    ON web.UserID = store.UserID
;

-- combine purchases online with purchases in store for users who visited only in store at least once
SELECT 	store.UserID AS UserID, 
		COALESCE(Purchases_Online, 0) AS Purchases_Online,
        COALESCE(Visits_Online, 0) AS Visits_Online,
        COALESCE(Purchases_Store, 0) AS Purchases_Store,
        COALESCE(Visits_Store, 0) AS Visits_Store
FROM
(
	SELECT UserID, ROUND(SUM(PurchaseAmount),2) AS Purchases_Online, COUNT(PurchaseAmount) AS Visits_Online
	FROM web
	GROUP BY UserID
) AS web
RIGHT JOIN 
	(
		SELECT UserID, ROUND(SUM(PurchaseAmount),2) AS Purchases_Store, COUNT(PurchaseAmount) AS Visits_Store
		FROM store
		GROUP BY UserID
	) AS store
    ON web.UserID = store.UserID
;

-- combine purchases online with purchases in store for all users including those who visited only in store at least once
-- or only online at least once

SELECT 	web.UserID AS UserID, 
		COALESCE(Purchases_Online, 0) AS Purchases_Online,
        COALESCE(Visits_Online, 0) AS Visits_Online,
        COALESCE(Purchases_Store, 0) AS Purchases_Store,
        COALESCE(Visits_Store, 0) AS Visits_Store
FROM
(
	SELECT UserID, ROUND(SUM(PurchaseAmount),2) AS Purchases_Online, COUNT(PurchaseAmount) AS Visits_Online
	FROM web
	GROUP BY UserID
) AS web
LEFT JOIN 
	(
		SELECT UserID, ROUND(SUM(PurchaseAmount),2) AS Purchases_Store, COUNT(PurchaseAmount) AS Visits_Store
		FROM store
		GROUP BY UserID
	) AS store
    ON web.UserID = store.UserID
UNION
SELECT 	store.UserID AS UserID, 
		COALESCE(Purchases_Online, 0) AS Purchases_Online,
        COALESCE(Visits_Online, 0) AS Visits_Online,
        COALESCE(Purchases_Store, 0) AS Purchases_Store,
        COALESCE(Visits_Store, 0) AS Visits_Store
FROM
(
	SELECT UserID, ROUND(SUM(PurchaseAmount),2) AS Purchases_Online, COUNT(PurchaseAmount) AS Visits_Online
	FROM web
	GROUP BY UserID
) AS web
RIGHT JOIN 
	(
		SELECT UserID, ROUND(SUM(PurchaseAmount),2) AS Purchases_Store, COUNT(PurchaseAmount) AS Visits_Store
		FROM store
		GROUP BY UserID
	) AS store
    ON web.UserID = store.UserID
ORDER BY UserID;

-- create a temporary data warehouse
DROP TABLE IF EXISTS salesdw_temp;

CREATE TABLE salesdw_temp AS
(SELECT 	web.UserID, 
		COALESCE(Purchases_Online, 0) AS Purchases_Online,
        COALESCE(Visits_Online, 0) AS Visits_Online,
        COALESCE(Purchases_Store, 0) AS Purchases_Store,
        COALESCE(Visits_Store, 0) AS Visits_Store
FROM
(
	SELECT UserID, ROUND(SUM(PurchaseAmount),2) AS Purchases_Online, COUNT(PurchaseAmount) AS Visits_Online
	FROM web
	GROUP BY UserID
) AS web
LEFT JOIN 
	(
		SELECT UserID, ROUND(SUM(PurchaseAmount),2) AS Purchases_Store, COUNT(PurchaseAmount) AS Visits_Store
		FROM store
		GROUP BY UserID
	) AS store
    ON web.UserID = store.UserID
UNION
SELECT 	store.UserID AS UserID, 
		COALESCE(Purchases_Online, 0) AS Purchases_Online,
        COALESCE(Visits_Online, 0) AS Visits_Online,
        COALESCE(Purchases_Store, 0) AS Purchases_Store,
        COALESCE(Visits_Store, 0) AS Visits_Store
FROM
(
	SELECT UserID, ROUND(SUM(PurchaseAmount),2) AS Purchases_Online, COUNT(PurchaseAmount) AS Visits_Online
	FROM web
	GROUP BY UserID
) AS web
RIGHT JOIN 
	(
		SELECT UserID, ROUND(SUM(PurchaseAmount),2) AS Purchases_Store, COUNT(PurchaseAmount) AS Visits_Store
		FROM store
		GROUP BY UserID
	) AS store
    ON web.UserID = store.UserID
ORDER BY UserID);

SELECT * FROM salesdw_temp;


SELECT * FROM newsletter;

-- add newsletter
SELECT *, CASE
			WHEN Newsletter.UserID IS NULL THEN 0
            ELSE 1
		END AS Newsletter
FROM salesdw_temp temp
	LEFT JOIN Newsletter ON temp.UserID = Newsletter.UserID
;

CREATE TABLE salesdw_wide AS
SELECT temp.UserID, Purchases_Online, Visits_Online, Purchases_Store, Visits_Store,
		CASE
			WHEN Newsletter.UserID IS NULL THEN 0
            ELSE 1
		END AS Newsletter
FROM salesdw_temp AS temp
	LEFT JOIN Newsletter ON temp.UserID = Newsletter.UserID
;

SELECT * FROM salesdw_wide;


DROP TABLE salesdw_temp;

-- statistics by newsletter that one can run on data warehouse
SELECT 	Newsletter,
		SUM(Purchases_Online) AS Online_Total, SUM(Visits_Online) AS Online_Visits, SUM(Purchases_Online)/SUM(Visits_Online) AS Online_Avg,
		SUM(Purchases_Store) AS Store_Total, SUM(Visits_Store) AS Store_Visits, SUM(Purchases_Store)/SUM(Visits_Store) AS Store_Avg
FROM salesdw_wide
GROUP BY Newsletter WITH ROLLUP;

-- count frequencies for visits online
SELECT Visits_Online, COUNT(*) AS Freq
FROM salesdw_wide
GROUP BY Visits_Online WITH ROLLUP;

-- count frequencies for visits online and visits in store
SELECT Visits_Online, Visits_Store, COUNT(*) AS Freq
FROM salesdw_wide
GROUP BY Visits_Online, Visits_Store WITH ROLLUP;




-- alternate way
SELECT * FROM newsletter;
SELECT * FROM web;
SELECT * FROM store;

-- combine purchases online and in store
SELECT UserID, PurchaseAmount AS Purchase_Online, 0 AS Purchase_Store
FROM web
UNION
SELECT UserID, 0 AS Purchase_Online, PurchaseAmount AS Purchase_Store
FROM store
ORDER BY UserID, Purchase_Online DESC, Purchase_Store DESC;

SELECT 	UserID,
		ROUND(SUM(Purchase_Online),2) AS Purchases_Online, 
		SUM(CASE WHEN Purchase_Online > 0 THEN 1 ELSE 0 END) AS Visits_Online,
		ROUND(SUM(Purchase_Store),2) AS Purchases_Store, 
        SUM(CASE WHEN Purchase_Store > 0 THEN 1 ELSE 0 END) AS Visits_Store
FROM
(
	SELECT UserID, PurchaseAmount AS Purchase_Online, 0 AS Purchase_Store
	FROM web
	UNION
	SELECT UserID, 0 AS Purchase_Online, PurchaseAmount AS Purchase_Store
	FROM store
) AS u
GROUP BY UserID
ORDER BY UserID;

-- combine with newsletter

SELECT summary.UserID, Purchases_Online, Visits_Online, Purchases_Store, Visits_Store,
		CASE
			WHEN Newsletter.UserID IS NULL THEN 0
            ELSE 1
		END AS Newsletter
FROM
(
	SELECT UserID, ROUND(SUM(Purchase_Online),2) AS Purchases_Online, SUM(CASE WHEN Purchase_Online > 0 THEN 1 ELSE 0 END) AS Visits_Online,
			   ROUND(SUM(Purchase_Store),2) AS Purchases_Store, SUM(CASE WHEN Purchase_Store > 0 THEN 1 ELSE 0 END) AS Visits_Store
	FROM
	(
		SELECT UserID, PurchaseAmount AS Purchase_Online, 0 AS Purchase_Store
		FROM web
		UNION
		SELECT UserID, 0 AS Purchase_Online, PurchaseAmount AS Purchase_Store
		FROM store
	) AS u
	GROUP BY UserID
) AS summary
LEFT JOIN Newsletter ON summary.UserID = Newsletter.UserID
ORDER BY summary.UserID;






