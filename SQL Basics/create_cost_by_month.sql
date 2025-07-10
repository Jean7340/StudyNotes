DROP DATABASE IF EXISTS cost_by_month;
CREATE DATABASE cost_by_month;

USE cost_by_month;
-- -----------------------------------------------------
-- Table `cost_by_month`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `cost_by_month` (
  `Department` VARCHAR(10) NULL DEFAULT NULL,
  `Manager` VARCHAR(100) NULL DEFAULT NULL,
  `Cost_Center` VARCHAR(45) NULL DEFAULT NULL,
  `Jan` VARCHAR(45) NULL DEFAULT NULL,
  `Feb` VARCHAR(45) NULL DEFAULT NULL,
  `Mar` VARCHAR(45) NULL DEFAULT NULL,
  `Apr` VARCHAR(45) NULL DEFAULT NULL,
  `May` VARCHAR(45) NULL DEFAULT NULL,
  `Jun` VARCHAR(45) NULL DEFAULT NULL,
  `Jul` VARCHAR(45) NULL DEFAULT NULL,
  `Aug` VARCHAR(45) NULL DEFAULT NULL,
  `Sep` VARCHAR(45) NULL DEFAULT NULL,
  `Oct` VARCHAR(45) NULL DEFAULT NULL,
  `Nov` VARCHAR(45) NULL DEFAULT NULL,
  `Dec` VARCHAR(45) NULL DEFAULT NULL);



INSERT INTO cost_by_month VALUES
('A','Casey','115Q',Null,Null,Null,Null,1365,Null,Null,1338,1305,Null,Null,497)
;


INSERT INTO cost_by_month VALUES
('A','Casey','116V',Null,Null,Null,Null,1455,1485,Null,1482,Null,Null,499,Null),
('A','Casey','12N',Null,469,924,Null,Null,1473,Null,Null,1278,Null,Null,Null)
;

INSERT INTO cost_by_month VALUES
('A','Casey','130T',Null,Null,Null,Null,1221,Null,1257,Null,1371,Null,Null,Null),
('A','Casey','146W',455,Null,Null,Null,Null,1395,1482,1305,Null,856,Null,453),
('A','Casey','65W',Null,Null,Null,960,1248,Null,Null,1428,Null,Null,Null,Null)
;



SELECT * FROM cost_by_month;