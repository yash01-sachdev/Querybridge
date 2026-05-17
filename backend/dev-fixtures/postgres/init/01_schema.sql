CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL
);

CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    category TEXT NOT NULL,
    price NUMERIC(10, 2) NOT NULL
);

CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    product_id INTEGER NOT NULL REFERENCES products(id),
    amount NUMERIC(10, 2) NOT NULL,
    status TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL
);

CREATE TABLE subscriptions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    plan TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TIMESTAMP NOT NULL
);

INSERT INTO users (name, email, status, created_at) VALUES
    ('Alice', 'alice@example.com', 'active', '2026-01-01 09:00:00'),
    ('Bob', 'bob@example.com', 'active', '2026-01-02 10:00:00'),
    ('Charlie', 'charlie@example.com', 'inactive', '2026-01-03 11:00:00');

INSERT INTO products (title, category, price) VALUES
    ('Notebook Pro', 'electronics', 999.99),
    ('Travel Mug', 'lifestyle', 24.50),
    ('Desk Lamp', 'home', 58.00);

INSERT INTO orders (user_id, product_id, amount, status, created_at) VALUES
    (1, 1, 120.50, 'completed', '2026-02-01 12:00:00'),
    (2, 2, 60.00, 'pending', '2026-02-02 12:00:00'),
    (1, 3, 300.00, 'completed', '2026-02-03 12:00:00'),
    (3, 2, 90.00, 'cancelled', '2026-02-04 12:00:00');

INSERT INTO subscriptions (user_id, plan, status, started_at) VALUES
    (1, 'pro', 'active', '2026-01-05 08:00:00'),
    (2, 'starter', 'active', '2026-01-06 08:00:00'),
    (3, 'starter', 'cancelled', '2026-01-07 08:00:00');
