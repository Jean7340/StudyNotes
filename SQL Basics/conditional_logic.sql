USE ap;

#CASE function

SELECT * FROM terms;

SELECT invoice_number, terms_id,

	CASE WHEN terms_id = 1 THEN "Net due 10 days" 
		 WHEN terms_id = 2 THEN "Net due 20 days"
         WHEN terms_id = 3 THEN "Net due 30 days"
         WHEN terms_id = 4 THEN "Net due 60 days"
         WHEN terms_id = 5 THEN "Net due 90 days"
    END AS terms
FROM invoices;    

SELECT invoice_number, terms_id,

	CASE terms_id WHEN  1 THEN "Net due 10 days" 
				  WHEN  2 THEN "Net due 20 days"
				  WHEN  3 THEN "Net due 30 days"
				  WHEN  4 THEN "Net due 60 days"
				  WHEN  5 THEN "Net due 90 days"
    END AS terms
FROM invoices;    


#searched CASE

SELECT CURDATE();

SELECT invoice_number, invoice_total, invoice_date, invoice_due_date,
	DATEDIFF(CURDATE(), invoice_due_date) AS days_past_due,

	CASE WHEN DATEDIFF(CURDATE(), invoice_due_date) >1180 THEN "Over 1180 days past due"
		 WHEN DATEDIFF(CURDATE(), invoice_due_date) >0 AND DATEDIFF(CURDATE(), invoice_due_date) <=1180
         THEN "1 to 1180 days past due"
         ELSE "current"
    END AS invoice_status 
    
FROM invoices
WHERE invoice_total - payment_total - credit_total>0;


#IF function

SELECT vendor_name, vendor_city,
	IF(vendor_city = "Fresno", "Yes", "No") AS city_fresno
FROM vendors;


#IFNULL and COALESCE functions

SELECT invoice_number, invoice_date, payment_date,
	IFNULL(payment_date,"No Payment") AS new_payment_date
FROM invoices; 

SELECT invoice_number, invoice_date, payment_date,
	COALESCE(payment_date, "No Payment") AS new_payment_date
FROM invoices;





#create test
CREATE TABLE ap.test
(city varchar(100),
state varchar(100),
country varchar(100)
);

INSERT INTO ap.test VALUES 
("Springfield", "IL", "USA"),
("Las Vegas", "NV", "USA"),
(Null, "PA", "USA"),
(Null, Null, "USA"),
("New York City", Null, "USA"),
(Null, Null, Null),
(Null, "AZ", "USA"),
(Null, "WI", "USA");

SELECT * FROM test;


#COALESCE
SELECT city, state, country,
	COALESCE (city, state, country, "none")
FROM test;    