USE ordersdb;

SELECT Part_Number, Entry_Number, Parts_Ordered
FROM orders
Order BY Part_Number, Entry_Number;

SELECT * FROM orders;

SELECT * FROM shipments;

-- Parts ordered and shipped by Entry Number
SELECT orders.Entry_Number, orders.Parts_Ordered, shipments.Parts_Shipped
FROM orders
	LEFT JOIN shipments
		ON orders.Entry_Number = shipments.Entry_Number
ORDER BY orders.Entry_Number;

-- aggregate
SELECT orders.Entry_Number, Parts_Ordered, SUM(Parts_Shipped)
FROM orders
	LEFT JOIN shipments
		ON orders.Entry_Number = shipments.Entry_Number
GROUP BY orders.Entry_Number
ORDER BY orders.Entry_Number;

-- note: what your global variable for only_full_group_by is set to will dictate whether the above query runs
-- fix below is mostly for Mac users
SET sql_mode = (SELECT REPLACE(@@SQL_MODE, "ONLY_FULL_GROUP_BY", ""));




-- Parts Ordered & Parts Shipped by Part Number
SELECT orders.Part_Number, orders.Entry_Number, orders.Parts_Ordered, shipments.Parts_Shipped
FROM orders
	LEFT JOIN shipments
		ON orders.Entry_Number = shipments.Entry_Number
ORDER BY orders.Part_Number, orders.Entry_Number;


-- notice incorrect totals!
SELECT orders.Part_Number, orders.Entry_Number, SUM(orders.Parts_Ordered), SUM(shipments.Parts_Shipped)
FROM orders
	LEFT JOIN shipments
		ON orders.Entry_Number = shipments.Entry_Number
GROUP BY orders.Part_Number, orders.Entry_Number WITH ROLLUP;


SELECT *
FROM (
	(SELECT orders.Entry_Number, orders.Part_Number, orders.Parts_Ordered AS ordered, 0 AS shipped 
	FROM orders
	ORDER BY orders.Entry_Number)
UNION
	(SELECT shipments.Entry_Number, orders.Part_Number, 0 AS ordered, shipments.Parts_Shipped AS shipped
	FROM shipments JOIN orders
		ON shipments.Entry_Number = orders.Entry_Number
	ORDER BY shipments.Entry_Number)
    ) AS u
ORDER BY Part_Number, Entry_Number, ordered DESC
;

SELECT Part_Number, Entry_Number, SUM(ordered), SUM(shipped)
FROM (
(SELECT orders.Entry_Number, orders.Part_Number, orders.Parts_Ordered AS ordered, 0 AS shipped 
	FROM orders
	ORDER BY orders.Entry_Number)
UNION
	(SELECT shipments.Entry_Number, orders.Part_Number, 0 AS ordered, shipments.Parts_Shipped AS shipped
	FROM shipments JOIN orders
		ON shipments.Entry_Number = orders.Entry_Number
	ORDER BY shipments.Entry_Number)
) AS u
Group BY Part_Number, Entry_Number WITH ROLLUP;