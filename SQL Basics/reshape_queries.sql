USE cost_by_month;

SELECT * FROM cost_by_month;

-- wide to long

-- only Jan
SELECT Department, Manager, Cost_Center,
	SUM(Jan) AS Cost, "Jan" AS Month
FROM cost_by_month
GROUP BY
   Department, Manager, Cost_Center
ORDER BY
    Department, Manager, Cost_Center
;

SELECT Department, Manager, Cost_Center,
	SUM(Jan) AS Cost, "Jan" AS Month
FROM cost_by_month
GROUP BY
   Department, Manager, Cost_Center
HAVING SUM(Jan) IS NOT NULL
ORDER BY
    Department, Manager, Cost_Center;

-- three months
SELECT 
    Department,
    Manager,
    Cost_Center,
    SUM(Jan) AS Cost,
    'Jan' AS Month
FROM
    cost_by_month
GROUP BY Department , Manager , Cost_Center 
HAVING SUM(Jan) IS NOT NULL
UNION SELECT 
    Department,
    Manager,
    Cost_Center,
    SUM(Feb) AS Cost,
    'Feb' AS Month
FROM
    cost_by_month
GROUP BY Department , Manager , Cost_Center
HAVING SUM(Feb) IS NOT NULL 
UNION SELECT 
    Department,
    Manager,
    Cost_Center,
    SUM(Mar) AS Cost,
    'Mar' AS Month
FROM
    cost_by_month
GROUP BY Department , Manager , Cost_Center
HAVING SUM(Mar) IS NOT NULL
ORDER BY Department , Manager , Cost_Center
;



-- long to wide
DROP TABLE IF EXISTS cost_long;

CREATE TABLE cost_long AS
SELECT Department, Manager, Cost_Center,
	"Jan" AS Month, SUM(Jan) AS Cost
FROM cost_by_month
GROUP BY
   Department, Manager, Cost_Center
HAVING SUM(Jan) IS NOT NULL
UNION
SELECT Department, Manager, Cost_Center,
	"Feb" AS Month, SUM(Feb) AS Cost
FROM cost_by_month
GROUP BY
   Department, Manager, Cost_Center
HAVING SUM(Feb) IS NOT NULL
UNION
   SELECT Department, Manager, Cost_Center,
	"Mar" AS Month, SUM(Mar) AS Cost
FROM cost_by_month
GROUP BY
   Department, Manager, Cost_Center
HAVING SUM(Mar) IS NOT NULL
ORDER BY 
	Department, Manager, Cost_Center
;


SELECT * FROM cost_long;

SELECT 
   Department, Manager, Cost_Center,
   MAX(CASE WHEN Month = "Jan" THEN Cost END) AS Jan,
   MAX(CASE WHEN Month = "Feb" THEN Cost END) AS Feb,
   MAX(CASE WHEN Month = "Mar" THEN Cost END) AS Mar
FROM 
 cost_long
GROUP BY
   Department, Manager, Cost_Center
ORDER BY
    Department, Manager, Cost_Center
;


