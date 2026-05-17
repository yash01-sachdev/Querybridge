db = db.getSiblingDB("app");

db.users.deleteMany({});
db.orders.deleteMany({});

db.users.insertMany([
  {
    name: "Alice",
    email: "alice@example.com",
    status: "active",
    created_at: "2026-01-01T09:00:00Z",
  },
  {
    name: "Bob",
    email: "bob@example.com",
    status: "active",
    created_at: "2026-01-02T10:00:00Z",
  },
  {
    name: "Charlie",
    email: "charlie@example.com",
    status: "inactive",
    created_at: "2026-01-03T11:00:00Z",
  },
]);

db.orders.insertMany([
  {
    user_name: "Alice",
    amount: 120.5,
    status: "completed",
    created_at: "2026-02-01T12:00:00Z",
  },
  {
    user_name: "Bob",
    amount: 60.0,
    status: "pending",
    created_at: "2026-02-02T12:00:00Z",
  },
  {
    user_name: "Alice",
    amount: 300.0,
    status: "completed",
    created_at: "2026-02-03T12:00:00Z",
  },
  {
    user_name: "Charlie",
    amount: 90.0,
    status: "cancelled",
    created_at: "2026-02-04T12:00:00Z",
  },
]);
