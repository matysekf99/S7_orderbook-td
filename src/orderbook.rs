// ============================================================================
// REFERENCE IMPLEMENTATION (Naive, for correctness comparison)
// ============================================================================

use std::collections::BTreeMap;
use crate::interfaces::{OrderBook, Price, Quantity, Side, Update};

pub struct OrderBookImpl {
    bids: BTreeMap<Price, Quantity>,
    asks: BTreeMap<Price, Quantity>,
    best_bid: Option<Price>,
    best_ask: Option<Price>,
}

impl OrderBook for OrderBookImpl {
    fn new() -> Self {
        Self { 
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            best_bid: None,
            best_ask: None,
        }
    }

    fn apply_update(&mut self, update: Update) {
        match update {
            Update::Set { price, quantity, side } => {
                if quantity == 0 {
                    match side {
                        Side::Ask => {
                            self.asks.remove(&price);
                            if self.best_ask == Some(price) {
                                self.best_ask = self.asks.first_key_value().map(|(p, _)| *p);
                            }
                        },
                        Side::Bid => {
                            self.bids.remove(&price);
                            if self.best_bid == Some(price) {
                                self.best_bid = self.bids.last_key_value().map(|(p, _)| *p);
                            }
                        }
                    }
                } else {
                    match side {
                        Side::Ask => {
                            self.asks.insert(price, quantity);
                            if self.best_ask.is_none() || price < self.best_ask.unwrap() {
                                self.best_ask = Some(price);
                            }
                        },
                        Side::Bid => {
                            self.bids.insert(price, quantity);
                            if self.best_bid.is_none() || price > self.best_bid.unwrap() {
                                self.best_bid = Some(price);
                            }
                        }
                    }
                }
            },
            Update::Remove { price, side } => {
                match side {
                    Side::Ask => {
                        self.asks.remove(&price);
                        if self.best_ask == Some(price) {
                            self.best_ask = self.asks.first_key_value().map(|(p, _)| *p);
                        }
                    },
                    Side::Bid => {
                        self.bids.remove(&price);
                        if self.best_bid == Some(price) {
                            self.best_bid = self.bids.last_key_value().map(|(p, _)| *p);
                        }
                    }
                }
            }
        }
    }

    fn get_spread(&self) -> Option<Price> {
        Some(self.best_ask? - self.best_bid?)
    }

    fn get_best_bid(&self) -> Option<Price> {
        self.best_bid
    }

    fn get_best_ask(&self) -> Option<Price> {
        self.best_ask
    }

    fn get_quantity_at(&self, price: Price, side: Side) -> Option<Quantity> {
        match side {
            Side::Bid => self.bids.get(&price).copied(),
            Side::Ask => self.asks.get(&price).copied(),
        }
    }

    fn get_top_levels(&self, side: Side, n: usize) -> Vec<(Price, Quantity)> {
        match side {
            Side::Bid => {
                self.bids
                    .iter()
                    .rev()  
                    .take(n)
                    .map(|(p, q)| (*p, *q))
                    .collect()
            }
            Side::Ask => {
                self.asks
                    .iter()
                    .take(n)
                    .map(|(p, q)| (*p, *q))
                    .collect()
            }
        }
    }

    fn get_total_quantity(&self, side: Side) -> Quantity {
        if side == Side::Bid {
            self.bids.values().sum()
        } else {
            self.asks.values().sum()
        }
    }
}