USE onlineretail;

SELECT DISTINCT InvoiceNo FROM sales;
SELECT DISTINCT CustomerID FROM sales;

SELECT COUNT(*) FROM sales;

SELECT * FROM sales;

-- Add TotalAmount calculation
SELECT InvoiceNo, STR_TO_DATE(InvoiceDate, "%m/%d/%Y") AS Date, CustomerID, Country, Description,
		Quantity, UnitPrice, ROUND((Quantity * UnitPrice),2) AS TotalAmount
FROM sales
ORDER BY InvoiceNo, Date;



-- Summary: Total by Invoice
SELECT InvoiceNo, STR_TO_DATE(InvoiceDate, "%m/%d/%Y") AS Date, CustomerID, Country,
SUM(ROUND((Quantity * UnitPrice),2)) AS TotalAmount
FROM sales
GROUP BY InvoiceNo,Date,CustomerID,Country
ORDER BY InvoiceNo, Date;



-- Summary: Total by Customer
SELECT CustomerID,Country,(ROUND(SUM(TotalAmount),2)) AS CustomerTotal
FROM (
	SELECT InvoiceNo, STR_TO_DATE(InvoiceDate, "%m/%d/%Y") AS Date, CustomerID,
    Country,SUM(ROUND((Quantity * UnitPrice),2)) AS TotalAmount
	FROM sales
	GROUP BY InvoiceNo,Date,CustomerID,Country
     ) AS InvoiceTotals
GROUP BY CustomerID,Country
ORDER BY CustomerID;




-- Add Derived Fields to Customer Summary
SELECT CustomerID, Country, 
		SUM(TotalAmount) AS CustomerTotal, 
		AVG(TotalAmount) AS CustomerAverage, 
        MAX(Date) AS LastDate, 
        DATEDIFF("2018-01-01", MAX(Date)) AS DaysSinceLast, 
        COUNT(InvoiceNo) AS CustomerVolume
FROM (
		SELECT InvoiceNo, STR_TO_DATE(InvoiceDate, "%m/%d/%Y") AS Date, CustomerID, Country,
				SUM(ROUND((Quantity * UnitPrice),2)) AS TotalAmount
		FROM sales
		GROUP BY InvoiceNo
	) AS InvoiceTotals
GROUP BY CustomerID, Country
ORDER BY CustomerID;

SET sql_mode = (SELECT REPLACE(@@SQL_MODE, "ONLY_FULL_GROUP_BY", ""));





-- AddRFM Scores
SELECT CustomerID, Country, 
		SUM(TotalAMount) AS CustomerTotal, 
		Avg(TotalAmount) AS CustomerAverage, 
        DATEDIFF("2018-01-01", MAX(Date)) AS DaysSinceLast, 
        COUNT(InvoiceNo) AS CustomerVolume,
        NTILE(5) OVER (ORDER BY Avg(TotalAmount) DESC) AS AvgGroup,
        NTILE(5) OVER (ORDER BY DATEDIFF("2018-01-01", MAX(Date))) AS LastGroup,
        NTILE(5) OVER (ORDER BY COUNT(InvoiceNo) DESC) AS VolumeGroup
FROM (
		SELECT InvoiceNo, STR_TO_DATE(InvoiceDate, "%m/%d/%Y") AS Date, CustomerID, Country,
				SUM(ROUND((Quantity * UnitPrice),2)) AS TotalAmount
		FROM sales
		GROUP BY InvoiceNo
	) AS InvoiceTotals
GROUP BY CustomerID
ORDER BY CustomerID;


-- sort by Customer Average...
SELECT CustomerID, Country, SUM(TotalAMount) AS CustomerTotal, 
		Avg(TotalAmount) AS CustomerAverage, DATEDIFF("2018-01-01", MAX(Date)) AS DaysSinceLast, 
        COUNT(InvoiceNo) AS CustomerVolume,
        NTILE(5) OVER (ORDER BY Avg(TotalAmount) DESC) AS AvgGroup,
        NTILE(5) OVER (ORDER BY DATEDIFF("2018-01-01", MAX(Date))) AS LastGroup,
        NTILE(5) OVER (ORDER BY COUNT(InvoiceNo) DESC) AS VolumeGroup
FROM (
		SELECT InvoiceNo, STR_TO_DATE(InvoiceDate, "%m/%d/%Y") AS Date, CustomerID, Country,
				SUM(ROUND((Quantity * UnitPrice),2)) AS TotalAmount
		FROM sales
		GROUP BY InvoiceNo
	) AS InvoiceTotals
GROUP BY CustomerID
ORDER BY CustomerAverage DESC;


-- Freq table for ntile
SELECT AvgGroup, COUNT(*) AS GroupSize
FROM
(
	SELECT CustomerID, Country, SUM(TotalAMount) AS CustomerTotal, 
		Avg(TotalAmount) AS CustomerAverage, DATEDIFF("2018-01-01", MAX(Date)) AS DaysSinceLast, 
        COUNT(InvoiceNo) AS CustomerVolume,
        NTILE(5) OVER (ORDER BY Avg(TotalAmount) DESC) AS AvgGroup,
        NTILE(5) OVER (ORDER BY DATEDIFF("2018-01-01", MAX(Date))) AS LastGroup,
        NTILE(5) OVER (ORDER BY COUNT(InvoiceNo) DESC) AS VolumeGroup
	FROM (
			SELECT InvoiceNo, STR_TO_DATE(InvoiceDate, "%m/%d/%Y") AS Date, CustomerID, Country,
					SUM(ROUND((Quantity * UnitPrice),2)) AS TotalAmount
			FROM sales
			GROUP BY InvoiceNo
			) AS InvoiceTotals
	GROUP BY CustomerID
) AS CustomerTotals
GROUP BY AvgGroup;
        

-- freq table for RFM scores
SELECT AvgGroup, LastGroup, VolumeGroup, COUNT(*) AS GroupSize
FROM
(
	SELECT CustomerID, Country, SUM(TotalAMount) AS CustomerTotal, 
		Avg(TotalAmount) AS CustomerAverage, DATEDIFF("2018-01-01", MAX(Date)) AS DaysSinceLast, 
        COUNT(InvoiceNo) AS CustomerVolume,
        NTILE(5) OVER (ORDER BY Avg(TotalAmount) DESC) AS AvgGroup,
        NTILE(5) OVER (ORDER BY DATEDIFF("2018-01-01", MAX(Date))) AS LastGroup,
        NTILE(5) OVER (ORDER BY COUNT(InvoiceNo) DESC) AS VolumeGroup
	FROM (
			SELECT InvoiceNo, STR_TO_DATE(InvoiceDate, "%m/%d/%Y") AS Date, CustomerID, Country,
					SUM(ROUND((Quantity * UnitPrice),2)) AS TotalAmount
			FROM sales
			GROUP BY InvoiceNo
			) AS InvoiceTotals
	GROUP BY CustomerID
) AS CustomerTotals
GROUP BY AvgGroup, LastGroup, VolumeGroup WITH ROLLUP;



CREATE TABLE rfmModel AS
SELECT CustomerID, Country, SUM(TotalAMount) AS CustomerTotal, 
		Avg(TotalAmount) AS CustomerAverage, DATEDIFF("2018-01-01", MAX(Date)) AS DaysSinceLast, 
        COUNT(InvoiceNo) AS CustomerVolume,
        NTILE(5) OVER (ORDER BY Avg(TotalAmount) DESC) AS AvgGroup,
        NTILE(5) OVER (ORDER BY DATEDIFF("2018-01-01", MAX(Date))) AS LastGroup,
        NTILE(5) OVER (ORDER BY COUNT(InvoiceNo) DESC) AS VolumeGroup
FROM (
		SELECT InvoiceNo, STR_TO_DATE(InvoiceDate, "%m/%d/%Y") AS Date, CustomerID, Country,
				SUM(ROUND((Quantity * UnitPrice),2)) AS TotalAmount
		FROM sales
		GROUP BY InvoiceNo
	) AS InvoiceTotals
GROUP BY CustomerID
ORDER BY CustomerAverage DESC;

SELECT * FROM rfmModel;

