import { MongoClient } from 'mongodb';
import { NextRequest, NextResponse } from 'next/server';

// Database connection
const uri = "mongodb://localhost:27017/";
const client = new MongoClient(uri);
const dbName = "restaurant_db";

export async function GET(request: NextRequest, { params }: { params: { customer_id: string } }) {
  try {
    // Connect to MongoDB
    await client.connect();
    console.log('Connected to MongoDB');
    
    const db = client.db(dbName);
    const collection = db.collection('customers');
    
    // If a specific customer_id is provided, return that customer's data
    if (params.customer_id && params.customer_id !== 'all') {
      const customer = await collection.findOne({ customerID: params.customer_id });
      
      if (!customer) {
        return NextResponse.json({ error: `Customer '${params.customer_id}' not found` }, { status: 404 });
      }
      
      return NextResponse.json(customer);
    }
    
    // If 'all' is specified, return all customer IDs
    if (params.customer_id === 'all') {
      const customers = await collection.find({}).project({ customerID: 1, _id: 0 }).toArray();
      return NextResponse.json(customers);
    }
    
    // Default response if no valid parameter is provided
    return NextResponse.json({ error: 'Invalid request' }, { status: 400 });
  } catch (error) {
    console.error('Error connecting to MongoDB:', error);
    return NextResponse.json({ error: 'Failed to connect to database' }, { status: 500 });
  } finally {
    // Close the connection when done
    await client.close();
    console.log('Disconnected from MongoDB');
  }
}