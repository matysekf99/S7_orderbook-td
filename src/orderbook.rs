// ============================================================================
// REFERENCE IMPLEMENTATION (Naive, for correctness comparison)
// ============================================================================

use std::collections::HashMap;
use itertools::Itertools;
use crate::interfaces::{OrderBook, Price, Quantity, Side, Update};

pub struct OrderBookImpl {
    bids:HashMap<Price,Quantity>,
    asks:HashMap<Price,Quantity>
}

impl OrderBook for OrderBookImpl {
    fn new() -> Self {
        Self { 
            bids: HashMap::new(),
            asks: HashMap::new()
        }
    }

    fn apply_update(&mut self, update: Update) {
        match update{
            Update::Set{ price, quantity, side } =>{
                if quantity==0{
                    match side{
                        Side::Ask=>{
                            self.asks.remove(&price);
                        },
                        Side::Bid=>{
                            self.bids.remove(&price);
                        }
                    }
                }
                else{
                    match side{
                        Side::Ask=>{
                            self.asks.insert(price, quantity);
                        },
                        Side::Bid=>{
                            self.bids.insert(price, quantity);
                        }
                    }
                }
            },
            Update::Remove{ price, side }=>{
                match side{
                    Side::Ask=>{
                        self.asks.remove(&price);
                    },
                    Side::Bid=>{
                        self.bids.remove(&price);
                    }
                }
            }
        }
    }

    fn get_spread(&self) -> Option<Price> {
        let min_price_ask =  match self.asks.keys().min(){
            Some(v)=>v,
            None=>return None
        };
        let max_price_bid = match self.bids.keys().max(){
            Some(v)=>v,
            None=>return None
        };
        let spread = min_price_ask - max_price_bid;
        Some(spread)
    }

    fn get_best_bid(&self) -> Option<Price> {
        let best_bid =  match self.bids.keys().max(){
            Some(v)=>v,
            None=>return None
        };
        Some(*best_bid)
    }

    fn get_best_ask(&self) -> Option<Price> {
        let best_ask =  match self.asks.keys().min(){
            Some(v)=>v,
            None=>return None
        };
        Some(*best_ask)
    }

    fn get_quantity_at(&self, price: Price, side: Side) -> Option<Quantity> {
        match side{
            Side::Bid=>{
                if self.bids.contains_key(&price){
                    Some(self.bids[&price])
                }
                else{
                    None
                }
            },
            Side::Ask=>{
                if self.asks.contains_key(&price){
                    Some(self.asks[&price])
                }
                else{
                    None
                }
            }
        }
    }

    fn get_top_levels(&self, side: Side, n: usize) -> Vec<(Price, Quantity)> {
        match side{
            Side::Bid => {
                self.bids
                    .iter()
                    .sorted_by(|a, b| b.0.cmp(&a.0))
                    .take(n)
                    .map(|(p, q)| (*p, *q))
                    .collect()
            }
            Side::Ask=>{
                self.asks
                    .iter()
                    .sorted_by(|a, b| a.0.cmp(&b.0))
                    .take(n)
                    .map(|(p, q)| (*p, *q))
                    .collect()
            }
        }
    }

    fn get_total_quantity(&self, side: Side) -> Quantity {
        if side == Side::Bid{
            self.bids.values().sum()
        }
        else {
            self.asks.values().sum()
        }
    }
}
