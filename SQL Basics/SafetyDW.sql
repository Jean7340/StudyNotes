USE safety;


-- Data Warehouse - kpis view
CREATE OR REPLACE VIEW safetykpis AS
SELECT locations.Location_ID, Headcount, 
		IF(Employee_Safety_Committee = "yes", 1, 0) AS committee,
		NMI, LTI, Last_Training, Last_Audit
FROM locations 
	JOIN 	(	SELECT 	Location_ID, COUNT(nmi.Incident_ID) AS NMI
				FROM nmi 
				GROUP BY Location_ID
			) AS NMI_query
		ON locations.Location_ID = NMI_query.Location_ID
	JOIN 	(	SELECT 	Location_ID, COUNT(lti.Incident_ID) AS LTI
				FROM lti 
				GROUP BY Location_ID
			) AS LTI_query
		ON locations.Location_ID = LTI_query.Location_ID
	JOIN 	(	SELECT 	Location_ID, 
						MAX(STR_TO_DATE(Training_Date, "%m/%d/%Y")) AS Last_Training
				FROM trainings
				GROUP BY Location_ID
			) AS last_training
		ON locations.Location_ID = last_training.Location_ID
	JOIN 	(	SELECT 	Location_ID, 
						MAX(STR_TO_DATE(Audit_Date, "%m/%d/%Y")) AS Last_Audit
				FROM audits
				GROUP BY Location_ID
			) AS last_audit
		ON locations.Location_ID = last_audit.Location_ID;
        
 
