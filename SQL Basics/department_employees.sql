-- example tables for chapter 4

USE ex;

CREATE TABLE departments
(
  department_number   INT           NOT NULL,
  department_name     VARCHAR(50)   NOT NULL,
  CONSTRAINT department_number_unq  
    UNIQUE (department_number)
);

INSERT INTO departments VALUES 
(1,'Accounting'),
(2,'Payroll'),
(3,'Operations'),
(4,'Personnel'),
(5,'Maintenance');

CREATE TABLE employees
(
  employee_id         INT               NOT NULL,
  last_name           VARCHAR(35)       NOT NULL,
  first_name          VARCHAR(35)       NOT NULL,
  department_number   INT               NOT NULL,
  manager_id          INT
);

INSERT INTO employees VALUES 
(1,'Smith','Cindy',2,null),
(2,'Jones','Elmer',4,1),
(3,'Simonian','Ralph',2,2),
(4,'Hernandez','Olivia',1,9),
(5,'Aaronsen','Robert',2,4),
(6,'Watson','Denise',6,8),
(7,'Hardy','Thomas',5,2),
(8,'O''Leary','Rhea',4,9),
(9,'Locario','Paulo',6,1);