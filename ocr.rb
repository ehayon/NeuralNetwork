require './NeuralNetwork'

net = NeuralNetwork.new(256, 25, 10, 0.2, 0.4)

training_data = [ [ [ 0 , 0 ] , [ 0 ] ] ,
                  [ [ 0 , 1 ] , [ 1 ] ] ,
                  [ [ 1 , 0 ] , [ 1 ] ] ,
                  [ [ 1 , 1 ] , [ 0 ] ] ]

matrix = IO.readlines('numbers.data').map { |line| line.split.map(&:to_i) }
training_data = []

matrix.each do |row|
  training_data << [row[0..255], row[-10..-1]]
end


300.times do 
  pp net.train(training_data.sample(30))
end


net.test(training_data.sample(20))

