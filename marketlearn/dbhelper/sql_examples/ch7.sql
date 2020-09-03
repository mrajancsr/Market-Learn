select 1 num, 'abc' str 
union 
select 9 num, 'xyz' str;


select a.cust_id, a.open_branch_id from account a
inner join branch b on b.branch_id = a.open_branch_id
where b.name = 'Woburn Branch'

select 'abcdefg', (chr(97), chr(98), chr(99), chr(100), chr(101), chr(102), chr(103)) as ascii_values;

select concat('danke sch', chr(111), 'n') as chk, emp_id from employee e2 
select 'danke sch' or char(111)

select length(char_fld) char_length from string_tbl

select overlay('goodbye  world' placing 'cruel' from 9 for 0)

select replace('goodbye world', 'goodbye', 'hello')

select substring('goodbye cruel world', 9, 5)

select substring(overlay('goodbye  world' placing 'cruel' from 9 for 0), 9, 5)

-- working with numeric data
select ceil(22.75), floor(22.75)
union all
select ceil(72.445), floor(72.445)
union all 
select round(22.75), round(72.445)
union all 
select round(72.4545, 3), round(22.090909, 3)
union all 
select trunc(72.0909,1), trunc(72.0909,2) -- doesn't round up or down
union all 
select trunc(17, -1), round(17,-1)

-- handling signed data
select account_id, sign(avail_balance), abs(avail_balance)  from account

-- handling dates
update account
set last_activity_date = 
(select max(t.txn_date) from "transaction" t 
where t.account_id = account_id)
where exists (select 1 from transaction t 
where t.account_id = account_id);
