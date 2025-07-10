DROP DATABASE IF EXISTS university; 
CREATE DATABASE university;

USE university;


CREATE TABLE students
(
	studentID				INT				PRIMARY KEY 	NOT NULL,
    studentFirstName		VARCHAR(40)		NOT NULL,
    studentLastName			VARCHAR(40)		NOT NULL
    );

CREATE TABLE courses
(
	courseID			VARCHAR(10)		NOT NULL	PRIMARY KEY,
    courseTitle			VARCHAR(60)		NOT NULL,
    creditHours			INT				NOT NULL,
    prereqCourse		VARCHAR(10)
    );
    
CREATE TABLE sections
(
	sectionNumber		INT				NOT NULL,
    courseID			VARCHAR(10)	 	NOT NULL,
    sectionDays			varchar(20)		NOT NULL,
    room				VARCHAR(20)		NOT NULL,
    facultyID			INT				NOT NULL,
    CONSTRAINT	sections_pk PRIMARY KEY (courseID, sectionNumber)
    );
    
CREATE TABLE professors
(
	facultyID				INT				PRIMARY KEY		NOT NULL,
    facultyFirstName		VARCHAR(40)		NOT NULL,
    facultyLastName			VARCHAR(40)		NOT NULL,
    department				VARCHAR(40)
    );
    
CREATE TABLE enrollment
(
	studentID			int				NOT NULL,
    courseID			VARCHAR(10)		NOT NULL,
    sectionNumber		int				NOT NULL,
    CONSTRAINT enrollment_pk	PRIMARY KEY (studentID, courseID, sectionNumber)
);

ALTER TABLE sections
ADD CONSTRAINT sections_fk_professors
FOREIGN KEY (facultyID) REFERENCES professors (facultyID);

ALTER TABLE sections
ADD CONSTRAINT sections_fk_courses
FOREIGN KEY (courseID) REFERENCES courses (courseID);

ALTER TABLE enrollment
ADD CONSTRAINT enrollment_fk_sections
FOREIGN KEY (courseID, sectionNumber) REFERENCES sections (courseID, sectionNumber);

ALTER TABLE enrollment
ADD CONSTRAINT enrollment_fk_students
FOREIGN KEY (studentID) REFERENCES students (studentID);

ALTER TABLE courses
ADD CONSTRAINT courses_fk_prereq
FOREIGN KEY (prereqCourse) REFERENCES courses (courseID);


