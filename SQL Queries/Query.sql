/*
 * Exploratory Data Analysis
 */

select order_hour_of_day, count(*) as total_orders
from orders
group by order_hour_of_day ; --orders over the day


select user_id, avg(days_since_prior_order) as average_days_between_orders, count(*) as total_orders
from orders
group by user_id; -- predict the number of days until the next order


select opp.product_id, p.product_name, count(*) as total_orders, 
p.aisle_id, p.department_id 
from order_products__prior opp 
left join products p
on opp.product_id = p.product_id
group by p.product_id
order by total_orders desc; -- show the most popular/sold items


select order_id, user_id, lead(days_since_prior_order) over (order by user_id asc) ,
row_number() over (order by user_id asc)
from orders; -- days until next order

SELECT order_id,p.aisle_id 
from order_products__prior opp 
join products p
on opp.product_id = p.product_id ; --exported to python to be used in a crosstab


select order_id,  p.department_id 
from order_products__prior opp 
left JOIN products p
on opp.product_id = p.product_id ; --exported to python to be used in a crosstab


/*
 * First exported dataset - missing many features and has aisle count - a feature that was later dropped
 */

SELECT o.order_id, user_id, 
order_number -1 as Previous_orders_number, 
lead(days_since_prior_order) over (order by user_id,order_number) as days_until_next_order,
avg(days_since_prior_order)
over (partition by user_id )as average_days_between_orders,
CASE 
	WHEN LEAD(user_id) over(order by user_id, order_number) = user_id THEN 1 
	ELSE 0 
END AS event,
oia.*
from orders o
left join order_ids_and_aisles oia
on oia.order_id = o.order_id  
where aisle_1 is not null
order by user_id,Previous_orders_number ; 

/*
 * Primary Query that created the dataset used in this project
 */

WITH basket_size_id AS (
	SELECT order_id,
	COUNT(product_id) AS basket_size,
	SUM(reordered) AS reorder_num
	FROM order_products__prior
	GROUP BY order_id),
base_orders AS (
    SELECT
    o.*,
    bs.basket_size,
    bs.reorder_num
	FROM orders o
	LEFT JOIN basket_size_id bs
	ON o.order_id = bs.order_id),
historical_features AS (
SELECT *,
    COUNT(days_since_prior_order) OVER(
        PARTITION BY user_id
        ORDER BY order_number
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS previous_interval_count,
    AVG(CAST(days_since_prior_order AS REAL)) OVER(
        PARTITION BY user_id
        ORDER BY order_number
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS previous_interval_average,
    AVG(CAST(days_since_prior_order AS REAL) *
        CAST(days_since_prior_order AS REAL)) OVER(
	        PARTITION BY user_id
	        ORDER BY order_number
	        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS previous_interval_squared_average
FROM base_orders),
historical_features_with_std AS (
    SELECT *,
     SQRT(previous_interval_squared_average -
     (previous_interval_average * previous_interval_average)) AS interval_std
    FROM historical_features),
features AS (
	SELECT user_id, order_id,
	order_number - 1 AS Previous_orders_number,
	LEAD(days_since_prior_order) OVER(
		PARTITION BY user_id
		ORDER BY order_number)AS days_until_next_order,
    days_since_prior_order,
    AVG(CAST(days_since_prior_order AS REAL)) OVER(
         PARTITION BY user_id
         ORDER BY order_number
         ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)AS average_days_between_orders,
    AVG(CAST(days_since_prior_order AS REAL)) OVER(
         PARTITION BY user_id
         ORDER BY order_number
         ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS average_of_3_last_orders,
    MAX(CAST(days_since_prior_order AS REAL)) OVER(
         PARTITION BY user_id
         ORDER BY order_number
         ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS max_interval,
    MIN(days_since_prior_order) OVER(
         PARTITION BY user_id
         ORDER BY order_number
         ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS min_interval,
	basket_size,
	AVG(basket_size) OVER(
         PARTITION BY user_id
         ORDER BY order_number
         ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS average_basket_size,
    CAST(basket_size AS REAL)/NULLIF(
        AVG(basket_size) OVER(
            PARTITION BY user_id
            ORDER BY order_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),0)AS basket_ratio,
	CAST(reorder_num AS REAL)/ NULLIF(basket_size,0)
    AS reorder_percentage,
    order_dow,
    order_hour_of_day,
    interval_std
FROM historical_features_with_std )
SELECT *
FROM features 
ORDER BY user_id, Previous_orders_number;






























